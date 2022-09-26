include ../support/Makefile.inc

.PHONY: build clean test

build: $(BIN)/$(HL_TARGET)/filter

$(GENERATOR_BIN)/simple_pipeline.generator: simple_pipeline_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/simple_pipeline.a: $(GENERATOR_BIN)/simple_pipeline.generator
	@mkdir -p $(@D)
	$^ -g simple_pipeline -e $(GENERATOR_OUTPUTS) -o $(@D) -f simple_pipeline target=$* auto_schedule=false

$(BIN)/%/simple_pipeline_auto_schedule.a: $(GENERATOR_BIN)/simple_pipeline.generator
	@mkdir -p $(@D)
	$^ -g simple_pipeline -e $(GENERATOR_OUTPUTS) -o $(@D) -f simple_pipeline_auto_schedule target=$*-no_runtime auto_schedule=true

$(BIN)/%/filter: filter.cpp $(BIN)/%/simple_pipeline.a $(BIN)/%/simple_pipeline_auto_schedule.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS)

$(BIN)/%/out.png: $(BIN)/%/process
	@mkdir -p $(@D)
	$< $(IMAGES)/rgb.png 10 $@

clean:
	rm -rf $(BIN)

test: $(BIN)/$(HL_TARGET)/out.png


BATCH_ID ?= 0
TRAIN_ONLY ?= 0
SAMPLES_DIR ?= autotuned_samples
autotune: $(GENERATOR_BIN)/simple_pipeline.generator $(AUTOSCHED_BIN)/featurization_to_sample $(AUTOSCHED_BIN)/retrain_cost_model $(AUTOSCHED_BIN)/libautoschedule_adams2019.so $(AUTOSCHED_SRC)/autotune_loop.sh
	HL_MACHINE_PARAMS=10,1,1 \
	SAMPLES_DIR=$(SAMPLES_DIR) \
	HL_PERMIT_FAILED_UNROLL=1 \
	bash $(AUTOSCHED_SRC)/autotune_loop.sh \
		$(GENERATOR_BIN)/simple_pipeline.generator \
		simple_pipeline \
		$(HL_TARGET) \
		$(AUTOSCHED_SRC)/baseline.weights \
		$(AUTOSCHED_BIN) \
		$(HALIDE_DISTRIB_PATH) \
		$(SAMPLES_DIR)