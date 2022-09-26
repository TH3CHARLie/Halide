include ../support/Makefile.inc


.PHONY: build clean test

build: $(BIN)/$(HL_TARGET)/process

$(GENERATOR_BIN)/lens_blur.generator: lens_blur_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/lens_blur.a: $(GENERATOR_BIN)/lens_blur.generator
	@mkdir -p $(@D)
	$^ -g lens_blur -e $(GENERATOR_OUTPUTS) -o $(@D) -f lens_blur target=$* auto_schedule=false

$(BIN)/%/lens_blur_auto_schedule.a: $(GENERATOR_BIN)/lens_blur.generator
	@mkdir -p $(@D)
	$^ -g lens_blur -e $(GENERATOR_OUTPUTS) -o $(@D) -f lens_blur_auto_schedule target=$*-no_runtime auto_schedule=true

$(BIN)/%/process: process.cpp $(BIN)/%/lens_blur.a $(BIN)/%/lens_blur_auto_schedule.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS)

$(BIN)/%/out.png: $(BIN)/%/process
	@mkdir -p $(@D)
	$< $(IMAGES)/rgb_small.png 32 13 0.5 32 3 $@

clean:
	rm -rf $(BIN)

test: $(BIN)/$(HL_TARGET)/out.png

BATCH_ID ?= 0
TRAIN_ONLY ?= 0
SAMPLES_DIR ?= autotuned_samples
autotune: $(GENERATOR_BIN)/lens_blur.generator $(AUTOSCHED_BIN)/featurization_to_sample $(AUTOSCHED_BIN)/retrain_cost_model $(AUTOSCHED_BIN)/libautoschedule_adams2019.so $(AUTOSCHED_SRC)/autotune_loop.sh
	HL_MACHINE_PARAMS=10,1,1 \
	SAMPLES_DIR=$(SAMPLES_DIR) \
	HL_PERMIT_FAILED_UNROLL=1 \
	bash $(AUTOSCHED_SRC)/autotune_loop.sh \
		$(GENERATOR_BIN)/lens_blur.generator \
		lens_blur \
		$(HL_TARGET) \
		$(AUTOSCHED_SRC)/baseline.weights \
		$(AUTOSCHED_BIN) \
		$(HALIDE_DISTRIB_PATH) \
		$(SAMPLES_DIR)