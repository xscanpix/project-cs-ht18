#!/usr/bin/make -f

PYTHON = python3
RUNMOVIDIUS = code/movidius/run.py
PREPAREMODEL = code/model/run.py
SETTINGS = settings.json
OUTPUTDIR = output

all: run_test

prepare: clean
	mkdir $(OUTPUTDIR)
	$(PYTHON) $(PREPAREMODEL) $(SETTINGS)	

run_test: prepare
	$(PYTHON) $(RUNMOVIDIUS) $(SETTINGS)

clean: 
	rm -rf $(OUTPUTDIR)
