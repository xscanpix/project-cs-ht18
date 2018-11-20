#!/usr/bin/make -f

PYTHON = python3
RUNFILE = run.py
SETTINGS = settings.json
OUTPUT = output
CONVERTFLAGS = -k2tf -f
COMPILEFLAGS = -c -f
CPUTESTFLAGS = -tmc -it 100
MOVIDIUSTESTFLAGS = -tmm -it 100
SETTINGSFLAGS = -s $(SETTINGS)


all: convert-compile test-cpu test-movidius


convert-keras: clean
	mkdir $(OUTPUT)
	$(PYTHON) $(RUNFILE) $(CONVERTFLAGS) $(SETTINGSFLAGS)


compile-graph:
	$(PYTHON) $(RUNFILE) $(COMPILEFLAGS) $(SETTINGSFLAGS)


convert-compile: convert-keras compile-graph


test-cpu:
	taskset --cpu-list 1,2 $(PYTHON) $(RUNFILE) $(CPUTESTFLAGS) $(SETTINGSFLAGS)


test-movidius:
	$(PYTHON) $(RUNFILE) $(MOVIDIUSTESTFLAGS) $(SETTINGSFLAGS)


clean: 
	rm -rf $(OUTPUT)
