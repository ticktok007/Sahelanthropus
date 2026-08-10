/*
 * cycle_counter.c — Weighted execution-cost QEMU plugin for RISC-V
 *
 * Counts dynamically executed RISC-V instructions and computes a
 * configurable weighted cost.
 *
 * Branch direction is NOT available via the public QEMU Plugin API
 * (exec callbacks fire with no way to determine taken/not-taken without
 * reading PC registers, which would require QEMU_PLUGIN_CB_R_REGS and
 * architecture-specific register enumeration).  All branches are therefore
 * classified as branch_unknown with weight 1.
 *
 * Load hit/miss detection uses a direct-mapped cache model driven by the
 * effective virtual address provided by the memory callback.
 *
 * Plugin arguments:
 *   output=<path>        Write JSON to a file (default: stdout)
 *   stdout=1             Force JSON to stdout even if output= is set
 *   cache_size=<bytes>   Total cache size (default: 4096)
 *   line_size=<bytes>    Cache line size  (default: 64)
 *
 * Build:  ./build_cycle_counter.sh
 * Run:    ./run_cycle_counter.sh <binary> [output.json]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <glib.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

/* ================================================================== */
/* Instruction category IDs                                             */
/* ================================================================== */
typedef enum {
    CAT_ALU = 0,
    CAT_MUL,
    CAT_DIV,
    CAT_LOAD,       /* resolved to hit/miss at runtime via mem callback */
    CAT_STORE,
    CAT_BRANCH,     /* direction unknown → branch_unknown, weight 1     */
    CAT_JUMP_CALL,
    CAT_RETURN,
    CAT_SYSTEM_CSR,
    CAT_ATOMIC,
    CAT_FP,
    CAT_VECTOR,
    CAT_OTHER,
} InsnCat;

/* ================================================================== */
/* Weights                                                              */
/* ================================================================== */
#define W_ALU           1
#define W_MUL           3
#define W_DIV           20
#define W_LOAD_HIT      1
#define W_LOAD_MISS     10
#define W_LOAD_UNKNOWN  10
#define W_STORE         1
#define W_BRANCH_TAKEN      5   /* never used; branch direction unknown */
#define W_BRANCH_NOT_TAKEN  1   /* never used; branch direction unknown */
#define W_BRANCH_UNKNOWN    1
#define W_JUMP_CALL     1
#define W_RETURN        1
#define W_SYSTEM_CSR    1
#define W_ATOMIC        1
#define W_FP            1
#define W_VECTOR        1
#define W_OTHER         1

/* ================================================================== */
/* Cache configuration (set from plugin args)                           */
/* ================================================================== */
static uint32_t  cfg_cache_size = 4096;
static uint32_t  cfg_line_size  = 64;
static uint32_t  num_lines;
static uint64_t *cache_tags;        /* direct-mapped: one tag per line  */
#define CACHE_EMPTY UINT64_MAX      /* sentinel for "line not loaded"   */

/* ================================================================== */
/* Global counters — protected by g_lock                               */
/* ================================================================== */
static GMutex g_lock;

static uint64_t g_weighted_cost     = 0;
static uint64_t g_total_insns       = 0;

static uint64_t g_cnt_alu           = 0;
static uint64_t g_cnt_mul           = 0;
static uint64_t g_cnt_div           = 0;
static uint64_t g_cnt_load_hit      = 0;
static uint64_t g_cnt_load_miss     = 0;
static uint64_t g_cnt_load_unknown  = 0;
static uint64_t g_cnt_store         = 0;
static uint64_t g_cnt_br_taken      = 0; /* always 0 — direction unknowable */
static uint64_t g_cnt_br_not_taken  = 0; /* always 0 — direction unknowable */
static uint64_t g_cnt_br_unknown    = 0;
static uint64_t g_cnt_jump_call     = 0;
static uint64_t g_cnt_return        = 0;
static uint64_t g_cnt_system_csr    = 0;
static uint64_t g_cnt_atomic        = 0;
static uint64_t g_cnt_fp            = 0;
static uint64_t g_cnt_vector        = 0;
static uint64_t g_cnt_other         = 0;

static uint64_t g_cache_hits        = 0;
static uint64_t g_cache_misses      = 0;
static uint64_t g_cache_addr_unk    = 0;

/* Opcode histogram: interned const char * → heap uint64_t * counter  */
static GHashTable *g_opcodes = NULL;

/* ================================================================== */
/* Plugin args                                                           */
/* ================================================================== */
static char *g_output_path  = NULL;
static bool  g_force_stdout = false;

/* ================================================================== */
/* Userdata encoding for insn exec callback                             */
/*                                                                      */
/* We pack the InsnCat (0-12, fits in 4 bits) into the low bits of the */
/* void* pointer.  Safe on LP64 because all function pointers and      */
/* global addresses are aligned.                                        */
/* ================================================================== */
#define PACK_CAT(c)   ((void *)(uintptr_t)(c))
#define UNPACK_CAT(p) ((InsnCat)((uintptr_t)(p) & 0x0F))

/* ================================================================== */
/* RISC-V 32-bit opcode constants                                       */
/* ================================================================== */
#define RV_OP_LOAD       0x03
#define RV_OP_LOAD_FP    0x07
#define RV_OP_MISC_MEM   0x0F
#define RV_OP_OP_IMM     0x13
#define RV_OP_AUIPC      0x17
#define RV_OP_OP_IMM_32  0x1B
#define RV_OP_STORE      0x23
#define RV_OP_STORE_FP   0x27
#define RV_OP_AMO        0x2F
#define RV_OP_OP         0x33
#define RV_OP_LUI        0x37
#define RV_OP_OP_32      0x3B
#define RV_OP_MADD       0x43
#define RV_OP_MSUB       0x47
#define RV_OP_NMSUB      0x4B
#define RV_OP_NMADD      0x4F
#define RV_OP_OP_FP      0x53
#define RV_OP_OP_V       0x57
#define RV_OP_BRANCH     0x63
#define RV_OP_JALR       0x67
#define RV_OP_JAL        0x6F
#define RV_OP_SYSTEM     0x73

/* ================================================================== */
/* Classify a 32-bit RISC-V instruction                                 */
/* ================================================================== */
static InsnCat classify_rv32(uint32_t raw)
{
    uint32_t op = raw & 0x7F;
    uint32_t f3 = (raw >> 12) & 0x7;
    uint32_t f7 = (raw >> 25) & 0x7F;

    switch (op) {
    case RV_OP_LOAD:      return CAT_LOAD;

    case RV_OP_LOAD_FP:
        /* funct3 6/7 = vector loads; others = FP loads */
        return (f3 >= 6) ? CAT_VECTOR : CAT_FP;

    case RV_OP_MISC_MEM:  return CAT_SYSTEM_CSR;  /* fence, fence.i */

    case RV_OP_OP_IMM:
    case RV_OP_OP_IMM_32:
    case RV_OP_LUI:
    case RV_OP_AUIPC:     return CAT_ALU;

    case RV_OP_STORE:     return CAT_STORE;

    case RV_OP_STORE_FP:
        return (f3 >= 6) ? CAT_VECTOR : CAT_FP;

    case RV_OP_AMO:       return CAT_ATOMIC;

    case RV_OP_OP:
    case RV_OP_OP_32:
        if (f7 == 0x01)   /* M-extension: mul* or div/rem* */
            return (f3 <= 3) ? CAT_MUL : CAT_DIV;
        return CAT_ALU;

    case RV_OP_MADD:
    case RV_OP_MSUB:
    case RV_OP_NMSUB:
    case RV_OP_NMADD:
    case RV_OP_OP_FP:     return CAT_FP;

    case RV_OP_OP_V:      return CAT_VECTOR;

    case RV_OP_BRANCH:    return CAT_BRANCH;

    case RV_OP_JALR: {
        uint32_t rd  = (raw >> 7)  & 0x1F;
        uint32_t rs1 = (raw >> 15) & 0x1F;
        /*
         * Canonical ret: jalr x0, ra, 0  (rd=x0, rs1=x1, imm=0).
         * Also classify any "jalr x0, rs, imm" (rd=0) as RETURN since it
         * discards the link — this covers jr pseudo-instructions.
         */
        return (rd == 0) ? CAT_RETURN : CAT_JUMP_CALL;
        (void)rs1;
    }

    case RV_OP_JAL:       return CAT_JUMP_CALL;

    case RV_OP_SYSTEM:    return CAT_SYSTEM_CSR;

    default:              return CAT_OTHER;
    }
}

/* ================================================================== */
/* Classify a 16-bit compressed RISC-V instruction                     */
/* ================================================================== */
static InsnCat classify_rv16(uint16_t c)
{
    uint16_t cop  = c & 0x3;
    uint16_t cf3  = (c >> 13) & 0x7;

    switch (cop) {
    case 0x0: /* Quadrant 0 */
        if (cf3 == 0x2 || cf3 == 0x3) return CAT_LOAD;   /* c.lw / c.ld  */
        if (cf3 == 0x6 || cf3 == 0x7) return CAT_STORE;  /* c.sw / c.sd  */
        return CAT_ALU;   /* c.addi4spn, c.addi etc.                      */

    case 0x1: /* Quadrant 1 */
        if (cf3 == 0x6 || cf3 == 0x7) return CAT_BRANCH; /* c.beqz/c.bnez */
        if (cf3 == 0x5)               return CAT_JUMP_CALL; /* c.j        */
        if (cf3 == 0x1)               return CAT_JUMP_CALL; /* c.jal(RV32)*/
        return CAT_ALU;

    case 0x2: /* Quadrant 2 */
        if (cf3 == 0x2 || cf3 == 0x3) return CAT_LOAD;   /* c.lwsp/c.ldsp */
        if (cf3 == 0x6 || cf3 == 0x7) return CAT_STORE;  /* c.swsp/c.sdsp */
        if (cf3 == 0x4) {
            uint16_t rs1  = (c >> 7) & 0x1F;
            uint16_t rs2  = (c >> 2) & 0x1F;
            bool     jbit = (c >> 12) & 0x1;
            if (!jbit && rs2 == 0 && rs1 != 0)
                return (rs1 == 1) ? CAT_RETURN : CAT_JUMP_CALL; /* c.jr */
            if (jbit && rs2 == 0 && rs1 != 0)
                return CAT_JUMP_CALL;   /* c.jalr */
            return CAT_ALU;             /* c.mv, c.add */
        }
        return CAT_ALU;

    default:
        return CAT_OTHER;
    }
}

/* ================================================================== */
/* Mnemonic extraction — first whitespace-delimited token, lowercased  */
/* ================================================================== */
static void extract_mnemonic(const char *disas, char *buf, size_t bufsz)
{
    if (!disas || !*disas) {
        g_strlcpy(buf, "unknown", bufsz);
        return;
    }
    size_t j = 0;
    for (size_t k = 0;
         disas[k] && !isspace((unsigned char)disas[k]) && j < bufsz - 1;
         k++)
    {
        unsigned char ch = (unsigned char)disas[k];
        if (isalnum(ch) || ch == '.' || ch == '_')
            buf[j++] = (char)tolower(ch);
    }
    if (j == 0)
        g_strlcpy(buf, "unknown", bufsz);
    else
        buf[j] = '\0';
}

/* ================================================================== */
/* Memory access callback — fires at exec time for every load/store    */
/* ================================================================== */
static void vcpu_mem_cb(unsigned int vcpu_index,
                        qemu_plugin_meminfo_t info,
                        uint64_t vaddr,
                        void *userdata)
{
    (void)vcpu_index;
    (void)userdata;

    /* We only registered for MEM_R, but double-check */
    if (qemu_plugin_mem_is_store(info)) return;

    g_mutex_lock(&g_lock);
    g_total_insns++;

    if (vaddr == 0) {
        /*
         * vaddr == 0 is theoretically valid but almost never for normal
         * loads.  Treat it as load_unknown to avoid spurious cache pollution.
         */
        g_cnt_load_unknown++;
        g_weighted_cost += W_LOAD_UNKNOWN;
        g_cache_addr_unk++;
    } else {
        uint64_t line_id = vaddr / cfg_line_size;
        uint32_t idx     = (uint32_t)(line_id % num_lines);
        if (cache_tags[idx] == line_id) {
            g_cnt_load_hit++;
            g_weighted_cost += W_LOAD_HIT;
            g_cache_hits++;
        } else {
            cache_tags[idx] = line_id;
            g_cnt_load_miss++;
            g_weighted_cost += W_LOAD_MISS;
            g_cache_misses++;
        }
    }
    g_mutex_unlock(&g_lock);
}

/* ================================================================== */
/* Instruction exec callback — fires for every non-load instruction    */
/* ================================================================== */
static void vcpu_insn_exec_cb(unsigned int vcpu_index, void *userdata)
{
    (void)vcpu_index;
    InsnCat cat = UNPACK_CAT(userdata);

    g_mutex_lock(&g_lock);
    g_total_insns++;

    switch (cat) {
    case CAT_ALU:        g_cnt_alu++;       g_weighted_cost += W_ALU;         break;
    case CAT_MUL:        g_cnt_mul++;       g_weighted_cost += W_MUL;         break;
    case CAT_DIV:        g_cnt_div++;       g_weighted_cost += W_DIV;         break;
    case CAT_STORE:      g_cnt_store++;     g_weighted_cost += W_STORE;       break;
    case CAT_BRANCH:
        /*
         * Branch direction (taken/not-taken) is not available via the
         * QEMU Plugin API without reading PC registers.  Classify as
         * branch_unknown with weight 1.
         */
        g_cnt_br_unknown++;
        g_weighted_cost += W_BRANCH_UNKNOWN;
        break;
    case CAT_JUMP_CALL:  g_cnt_jump_call++; g_weighted_cost += W_JUMP_CALL;  break;
    case CAT_RETURN:     g_cnt_return++;    g_weighted_cost += W_RETURN;      break;
    case CAT_SYSTEM_CSR: g_cnt_system_csr++;g_weighted_cost += W_SYSTEM_CSR; break;
    case CAT_ATOMIC:     g_cnt_atomic++;    g_weighted_cost += W_ATOMIC;      break;
    case CAT_FP:         g_cnt_fp++;        g_weighted_cost += W_FP;          break;
    case CAT_VECTOR:     g_cnt_vector++;    g_weighted_cost += W_VECTOR;      break;
    default:             g_cnt_other++;     g_weighted_cost += W_OTHER;       break;
    }
    g_mutex_unlock(&g_lock);
}

/* ================================================================== */
/* Opcode histogram exec callback — atomically bumps per-mnemonic ctr  */
/* ================================================================== */
static void vcpu_opcode_cb(unsigned int vcpu_index, void *userdata)
{
    (void)vcpu_index;
    /* userdata is a heap-allocated uint64_t* permanently owned by g_opcodes */
    __atomic_fetch_add((uint64_t *)userdata, 1ULL, __ATOMIC_RELAXED);
}

/* ================================================================== */
/* Translation callback — instruments every instruction in a TB        */
/* ================================================================== */
static void vcpu_tb_trans(struct qemu_plugin_tb *tb, void *userdata)
{
    (void)userdata;
    size_t n = qemu_plugin_tb_n_insns(tb);

    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);

        /* ---- Disassemble ---- */
        char *disas = qemu_plugin_insn_disas(insn);
        char mnem[64];
        extract_mnemonic(disas, mnem, sizeof(mnem));
        g_free(disas);

        /* ---- Get or create per-mnemonic counter ---- */
        g_mutex_lock(&g_lock);
        const char *key = g_intern_string(mnem);
        uint64_t   *ctr = (uint64_t *)g_hash_table_lookup(g_opcodes, key);
        if (!ctr) {
            ctr = g_new0(uint64_t, 1);
            g_hash_table_insert(g_opcodes, (gpointer)key, ctr);
        }
        g_mutex_unlock(&g_lock);

        /* ---- Decode instruction bytes ---- */
        size_t sz = qemu_plugin_insn_size(insn);
        uint32_t raw = 0;
        qemu_plugin_insn_data(insn, &raw, (sz < 4) ? sz : 4);

        /* ---- Classify ---- */
        InsnCat cat = CAT_OTHER;
        if (sz == 4) {
            cat = classify_rv32(raw);
        } else if (sz == 2) {
            cat = classify_rv16((uint16_t)(raw & 0xFFFF));
        }

        /* ---- Register exec callbacks ---- */
        if (cat == CAT_LOAD) {
            /*
             * Loads: total_insns and weighted_cost are bumped inside
             * vcpu_mem_cb which receives the effective virtual address.
             * We still need the opcode histogram bump — use vcpu_opcode_cb.
             */
            qemu_plugin_register_vcpu_mem_cb(insn, vcpu_mem_cb,
                                             QEMU_PLUGIN_CB_NO_REGS,
                                             QEMU_PLUGIN_MEM_R, NULL);
            qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_opcode_cb,
                                                   QEMU_PLUGIN_CB_NO_REGS,
                                                   (void *)ctr);
        } else {
            /* Category + cost counter */
            qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_insn_exec_cb,
                                                   QEMU_PLUGIN_CB_NO_REGS,
                                                   PACK_CAT(cat));
            /* Opcode histogram counter */
            qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_opcode_cb,
                                                   QEMU_PLUGIN_CB_NO_REGS,
                                                   (void *)ctr);
        }
    }
}

/* ================================================================== */
/* JSON output                                                           */
/* ================================================================== */
static void write_json(FILE *out)
{
    g_mutex_lock(&g_lock);

    /* Snapshot scalar counters */
    uint64_t wc    = g_weighted_cost;
    uint64_t tot   = g_total_insns;
    uint64_t c_alu = g_cnt_alu,  c_mul = g_cnt_mul,   c_div = g_cnt_div;
    uint64_t c_lhit= g_cnt_load_hit, c_lmiss= g_cnt_load_miss;
    uint64_t c_lunk= g_cnt_load_unknown;
    uint64_t c_sto = g_cnt_store;
    uint64_t c_bta = g_cnt_br_taken, c_bnt = g_cnt_br_not_taken;
    uint64_t c_buk = g_cnt_br_unknown;
    uint64_t c_jmp = g_cnt_jump_call, c_ret = g_cnt_return;
    uint64_t c_sys = g_cnt_system_csr, c_atm = g_cnt_atomic;
    uint64_t c_fp  = g_cnt_fp,   c_vec = g_cnt_vector, c_oth = g_cnt_other;
    uint64_t ca_h  = g_cache_hits, ca_m = g_cache_misses, ca_u = g_cache_addr_unk;

    /* Snapshot opcode key list (values read atomically after unlock) */
    GList *keys = g_hash_table_get_keys(g_opcodes);
    g_mutex_unlock(&g_lock);

    fprintf(out, "{\n");
    fprintf(out, "  \"weighted_cost\": %" PRIu64 ",\n", wc);
    fprintf(out, "  \"total_instructions\": %" PRIu64 ",\n", tot);

    /* weights */
    fprintf(out, "  \"weights\": {\n");
    fprintf(out, "    \"alu\": %d,\n",              W_ALU);
    fprintf(out, "    \"mul\": %d,\n",              W_MUL);
    fprintf(out, "    \"div\": %d,\n",              W_DIV);
    fprintf(out, "    \"load_hit\": %d,\n",         W_LOAD_HIT);
    fprintf(out, "    \"load_miss\": %d,\n",        W_LOAD_MISS);
    fprintf(out, "    \"load_unknown\": %d,\n",     W_LOAD_UNKNOWN);
    fprintf(out, "    \"store\": %d,\n",            W_STORE);
    fprintf(out, "    \"branch_taken\": %d,\n",     W_BRANCH_TAKEN);
    fprintf(out, "    \"branch_not_taken\": %d,\n", W_BRANCH_NOT_TAKEN);
    fprintf(out, "    \"branch_unknown\": %d,\n",   W_BRANCH_UNKNOWN);
    fprintf(out, "    \"jump_call\": %d,\n",        W_JUMP_CALL);
    fprintf(out, "    \"return\": %d,\n",           W_RETURN);
    fprintf(out, "    \"system_csr\": %d,\n",       W_SYSTEM_CSR);
    fprintf(out, "    \"atomic\": %d,\n",           W_ATOMIC);
    fprintf(out, "    \"floating_point\": %d,\n",   W_FP);
    fprintf(out, "    \"vector\": %d\n",            W_VECTOR);
    fprintf(out, "  },\n");

    /* counts */
    fprintf(out, "  \"counts\": {\n");
    fprintf(out, "    \"alu\": %" PRIu64 ",\n",              c_alu);
    fprintf(out, "    \"mul\": %" PRIu64 ",\n",              c_mul);
    fprintf(out, "    \"div\": %" PRIu64 ",\n",              c_div);
    fprintf(out, "    \"load_hit\": %" PRIu64 ",\n",         c_lhit);
    fprintf(out, "    \"load_miss\": %" PRIu64 ",\n",        c_lmiss);
    fprintf(out, "    \"load_unknown\": %" PRIu64 ",\n",     c_lunk);
    fprintf(out, "    \"store\": %" PRIu64 ",\n",            c_sto);
    fprintf(out, "    \"branch_taken\": %" PRIu64 ",\n",     c_bta);
    fprintf(out, "    \"branch_not_taken\": %" PRIu64 ",\n", c_bnt);
    fprintf(out, "    \"branch_unknown\": %" PRIu64 ",\n",   c_buk);
    fprintf(out, "    \"jump_call\": %" PRIu64 ",\n",        c_jmp);
    fprintf(out, "    \"return\": %" PRIu64 ",\n",           c_ret);
    fprintf(out, "    \"system_csr\": %" PRIu64 ",\n",       c_sys);
    fprintf(out, "    \"atomic\": %" PRIu64 ",\n",           c_atm);
    fprintf(out, "    \"floating_point\": %" PRIu64 ",\n",   c_fp);
    fprintf(out, "    \"vector\": %" PRIu64 ",\n",            c_vec);
    fprintf(out, "    \"other\": %" PRIu64 "\n",              c_oth);
    fprintf(out, "  },\n");

    /* cache */
    fprintf(out, "  \"cache\": {\n");
    fprintf(out, "    \"size_bytes\": %" PRIu32 ",\n",       cfg_cache_size);
    fprintf(out, "    \"line_size_bytes\": %" PRIu32 ",\n",  cfg_line_size);
    fprintf(out, "    \"lines\": %" PRIu32 ",\n",             num_lines);
    fprintf(out, "    \"hits\": %" PRIu64 ",\n",              ca_h);
    fprintf(out, "    \"misses\": %" PRIu64 ",\n",            ca_m);
    fprintf(out, "    \"unknown_address_loads\": %" PRIu64 "\n", ca_u);
    fprintf(out, "  },\n");

    /* opcodes */
    fprintf(out, "  \"opcodes\": {\n");
    for (GList *l = keys; l; l = l->next) {
        const char *k  = (const char *)l->data;
        uint64_t   *cp = (uint64_t *)g_hash_table_lookup(g_opcodes, k);
        uint64_t    cv = cp ? __atomic_load_n(cp, __ATOMIC_RELAXED) : 0;
        fprintf(out, "    \"%s\": %" PRIu64 "%s\n",
                k, cv, l->next ? "," : "");
    }
    g_list_free(keys);
    fprintf(out, "  }\n}\n");
}

/* ================================================================== */
/* atexit callback                                                       */
/* ================================================================== */
static void plugin_exit(void *userdata)
{
    (void)userdata;

    FILE *out       = NULL;
    bool  do_close  = false;

    if (g_force_stdout || !g_output_path) {
        out = stdout;
    } else {
        out = fopen(g_output_path, "w");
        if (!out) {
            fprintf(stderr,
                    "cycle_counter: cannot open '%s', writing to stdout\n",
                    g_output_path);
            out = stdout;
        } else {
            do_close = true;
        }
    }

    write_json(out);
    fflush(out);
    if (do_close) fclose(out);
}

/* ================================================================== */
/* Plugin entry point                                                    */
/* ================================================================== */
QEMU_PLUGIN_EXPORT
int qemu_plugin_install(qemu_plugin_id_t id,
                        const qemu_info_t *info,
                        int argc, char **argv)
{
    (void)info;

    for (int i = 0; i < argc; i++) {
        if (g_str_has_prefix(argv[i], "output=")) {
            g_free(g_output_path);
            g_output_path = g_strdup(argv[i] + 7);
        } else if (g_str_has_prefix(argv[i], "stdout=")) {
            g_force_stdout = (atoi(argv[i] + 7) != 0);
        } else if (g_str_has_prefix(argv[i], "cache_size=")) {
            uint64_t v = strtoull(argv[i] + 11, NULL, 10);
            if (v > 0 && v <= (1ULL << 30))
                cfg_cache_size = (uint32_t)v;
        } else if (g_str_has_prefix(argv[i], "line_size=")) {
            uint64_t v = strtoull(argv[i] + 10, NULL, 10);
            if (v > 0 && v <= 65536)
                cfg_line_size = (uint32_t)v;
        }
    }

    if (cfg_line_size > cfg_cache_size) cfg_line_size = cfg_cache_size;
    num_lines = cfg_cache_size / cfg_line_size;
    if (num_lines == 0) num_lines = 1;

    cache_tags = (uint64_t *)g_malloc(num_lines * sizeof(uint64_t));
    for (uint32_t i = 0; i < num_lines; i++)
        cache_tags[i] = CACHE_EMPTY;

    g_mutex_init(&g_lock);
    /* Key: permanent interned string.  Value: heap uint64_t (freed by g_free). */
    g_opcodes = g_hash_table_new_full(g_str_hash, g_str_equal, NULL, g_free);

    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans, NULL);
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    return 0;
}