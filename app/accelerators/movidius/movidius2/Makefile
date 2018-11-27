#!/usr/bin/make -f

PYTHON = python3 -W ignore
RUNFILE = run.py
SETTINGS = settings.json
OUTPUT = outputs
CONVERTFLAGS = -k2tf
COMPILEFLAGS = -c
CPUTESTFLAGS = -tmc -it 100
MOVIDIUSTESTFLAGS = -tmm -it 100
SETTINGSFLAGS = -s $(SETTINGS)

PERFFLAGS = taskset --cpu-list 0


all: convert-compile test-movidius


convert-keras: clean
	mkdir $(OUTPUT)
	$(PYTHON) $(RUNFILE) $(CONVERTFLAGS) $(SETTINGSFLAGS)


compile-graph:
	$(PYTHON) $(RUNFILE) $(COMPILEFLAGS) $(SETTINGSFLAGS)


convert-compile: convert-keras compile-graph


test-cpu:
	$(PERFFLAGS) $(PYTHON) $(RUNFILE) $(CPUTESTFLAGS) $(SETTINGSFLAGS)


test-movidius:
	$(PERFFLAGS) $(PYTHON) $(RUNFILE) $(MOVIDIUSTESTFLAGS) $(SETTINGSFLAGS)


clean: 
	rm -rf $(OUTPUT)
