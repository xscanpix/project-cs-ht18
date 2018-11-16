#!/usr/bin/make -f

PYTHON = python3
RUNMOVIDIUS = code/run.py
PREPAREMODEL = code/model/keras2graph.py
SETTINGS = settings.json
OUTPUTDIR = output

all: prepare run_test

prepare: clean
	mkdir $(OUTPUTDIR)
	$(PYTHON) $(PREPAREMODEL) $(SETTINGS)	

run_test: 
	$(PYTHON) $(RUNMOVIDIUS) $(SETTINGS)

clean: 
	rm -rf $(OUTPUTDIR)
