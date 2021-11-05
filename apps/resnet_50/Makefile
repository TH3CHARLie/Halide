include ../support/Makefile.inc

SEED = 123

.PHONY: build clean test

build: $(BIN)/$(HL_TARGET)/process

$(BIN)/%/pytorch_weights/ok:
	@mkdir -p $(@D)
	python3 load_weights.py $(@D)
	echo "ok" > $@

$(GENERATOR_BIN)/resnet50.generator: Resnet50Generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/resnet50.a: $(GENERATOR_BIN)/resnet50.generator
	@mkdir -p $(@D)
	$^ -g resnet50 -o $(@D) -f resnet50 target=$* auto_schedule=false

$(BIN)/%/process: process.cpp $(BIN)/%/resnet50.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS)

benchmark_and_validate: $(BIN)/$(HL_TARGET)/process $(BIN)/$(HL_TARGET)/pytorch_weights/ok
	$< 10 $* $(BIN)/$(HL_TARGET)/pytorch_weights/ $(SEED) $(BIN)/$(HL_TARGET)/res50gen_output.bin
	python3 validate_resnet50_output.py $(BIN)/$(HL_TARGET)/res50gen_output.bin $(SEED)

clean:
	rm -rf $(BIN)

test:
	@echo "Only currently testing that resnet builds"


BATCH_ID ?= 0
TRAIN_ONLY ?= 0
SAMPLES_DIR ?= autotuned_samples
autotune: $(GENERATOR_BIN)/resnet50.generator $(AUTOSCHED_BIN)/featurization_to_sample $(AUTOSCHED_BIN)/retrain_cost_model $(AUTOSCHED_BIN)/libautoschedule_adams2019.so $(AUTOSCHED_SRC)/autotune_loop.sh
	HL_MACHINE_PARAMS=20,1,1 \
	SAMPLES_DIR=$(SAMPLES_DIR) \
	HL_PERMIT_FAILED_UNROLL=1 \
	bash $(AUTOSCHED_SRC)/autotune_loop.sh \
		$(GENERATOR_BIN)/resnet50.generator \
		resnet50 \
		$(HL_TARGET) \
		$(AUTOSCHED_SRC)/baseline.weights \
		$(AUTOSCHED_BIN) \
		$(HALIDE_DISTRIB_PATH) \
		$(SAMPLES_DIR)