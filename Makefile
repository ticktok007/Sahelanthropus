CC = gcc
CFLAGS = -O3 -fPIC -shared -std=gnu11 -Wall -Wextra
QEMU_PATH = ./qemu
INCLUDES = -I$(QEMU_PATH)/include/qemu
GLIB_CFLAGS = $(shell pkg-config --cflags glib-2.0)
GLIB_LIBS = $(shell pkg-config --libs glib-2.0)

all: cycle_counter.so

cycle_counter.so: cycle_counter.c
	$(CC) $(CFLAGS) $(INCLUDES) $(GLIB_CFLAGS) $< -o $@ $(GLIB_LIBS)

clean:
	rm -f cycle_counter.so