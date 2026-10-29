CXX  = g++
NVCC = /usr/bin/nvcc

CXXFLAGS  = -Iinclude -MMD -MP
NVCCFLAGS = -Iinclude -MMD -MP
LDLIBS    = -lcublas -lcurl -lz

BUILD = build
LIB   = $(BUILD)/libMLlib.a

# ---------- library ----------

SRCS      = $(shell find src -name '*.cpp')
CUDA_SRCS = $(shell find src -name '*.cu')

OBJS      = $(SRCS:%.cpp=$(BUILD)/%.o)
CUDA_OBJS = $(CUDA_SRCS:%.cu=$(BUILD)/%.o)

all: lib

lib: $(LIB)

$(LIB): $(OBJS) $(CUDA_OBJS)
	ar rcs $@ $^

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ---------- examples ----------
# each examples/<name>/main.cpp builds to build/<name>

EXAMPLES     = $(notdir $(patsubst %/,%,$(dir $(wildcard examples/*/main.cpp))))
EXAMPLE_BINS = $(addprefix $(BUILD)/, $(EXAMPLES))

examples: $(EXAMPLE_BINS)

$(BUILD)/examples/%.o: examples/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -Iexamples/common -c $< -o $@

$(BUILD)/%: $(BUILD)/examples/%/main.o $(LIB)
	$(NVCC) -o $@ $< $(LIB) $(LDLIBS)

# make run-function_fit, make run-mnist, ...
run-%: $(BUILD)/%
	./$<

# ---------- tests ----------
# each tests/test_*.cpp builds to build/tests/test_*; make test runs them all

TESTS     = $(basename $(notdir $(wildcard tests/test_*.cpp)))
TEST_BINS = $(addprefix $(BUILD)/tests/, $(TESTS))

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "== $$t"; ./$$t || exit 1; done

$(BUILD)/tests/%: $(BUILD)/tests/%.o $(LIB)
	$(NVCC) -o $@ $< $(LIB) $(LDLIBS)

# ---------- install ----------

install: $(LIB)
	mkdir -p /usr/local/include/mllib /usr/local/lib
	cp -r include/. /usr/local/include/mllib/
	cp $(LIB) /usr/local/lib/libMLlib.a

uninstall:
	rm -rf /usr/local/include/mllib
	rm -f /usr/local/lib/libMLlib.a

clean:
	rm -rf $(BUILD)

.PHONY: all lib examples test install uninstall clean
.SECONDARY: $(EXAMPLES:%=$(BUILD)/examples/%/main.o) $(TEST_BINS:%=%.o)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
