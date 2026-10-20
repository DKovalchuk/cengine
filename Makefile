CC      := gcc
CFLAGS  := -std=c11 -Wall -Wextra -Wpedantic -O3 -march=native -mavx2 -mfma -fPIC -fopenmp -D_POSIX_C_SOURCE=200809L
LDFLAGS := -shared -lm -lpthread -fopenmp
ARFLAGS := rcs

# Directories
SRCDIR  := src
INCDIR  := include
BUILDDIR:= build
OBJDIR  := $(BUILDDIR)/obj
LIBDIR  := $(BUILDDIR)/lib

# Source files (auto-discover)
SRCS := $(wildcard $(SRCDIR)/core/*.c) \
        $(wildcard $(SRCDIR)/math/*.c) \
        $(wildcard $(SRCDIR)/nn/*.c) \
        $(wildcard $(SRCDIR)/autograd/*.c) \
        $(wildcard $(SRCDIR)/training/*.c) \
        $(wildcard $(SRCDIR)/data/*.c) \
        $(wildcard $(SRCDIR)/parallel/*.c) \
        $(wildcard $(SRCDIR)/telemetry/*.c)

OBJS := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))

# Targets
SHARED_LIB := $(LIBDIR)/libcengine.so
STATIC_LIB := $(LIBDIR)/libcengine.a

# Test sources
TEST_SRCS := $(wildcard tests/test_*.c)
TEST_BINS := $(patsubst tests/%.c,$(BUILDDIR)/tests/%,$(TEST_SRCS))

.PHONY: all shared static clean test

all: shared static

shared: $(SHARED_LIB)

static: $(STATIC_LIB)

$(SHARED_LIB): $(OBJS) | $(LIBDIR)
	$(CC) $(LDFLAGS) -o $@ $^

$(STATIC_LIB): $(OBJS) | $(LIBDIR)
	$(AR) $(ARFLAGS) $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCDIR) -MMD -c -o $@ $<

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

# Tests
test: $(TEST_BINS)
	@echo "=== Running tests ==="
	@for t in $(TEST_BINS); do \
		echo ">>> $$t"; \
		$$t || exit 1; \
	done
	@echo "=== All tests passed ==="

$(BUILDDIR)/tests/%: tests/%.c $(STATIC_LIB) | $(BUILDDIR)/tests
	$(CC) $(CFLAGS) -I$(INCDIR) -o $@ $< $(STATIC_LIB) -lm -lpthread

$(BUILDDIR)/tests:
	mkdir -p $(BUILDDIR)/tests

clean:
	rm -rf $(BUILDDIR)

# Dependency tracking
-include $(OBJS:.o=.d)
