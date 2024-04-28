PROJECT = ddpg
SOURCES = $(wildcard *.py)
SPECIAL = Makefile README.md LICENSE
HOST = artamonovgi@192.168.1.10
REMOTE_PATH = /home/artamonovgi/my/Pursuit-Evasion-Game

finetune:
	cd Robot-Control && $(MAKE) graphic_run &
	./main.py --mode=train --debug --train_resume=0

train:
	cd Robot-Control && $(MAKE) graphic_run &
	./main.py --mode=train --debug

test:
	cd Robot-Control && $(MAKE) graphic_run &
	./main.py --mode=test --debug

cleanup:
	rm -r output/pursuit-run*

pull:
	rsync -rvza -e 'ssh -p 503' $(HOST):$(REMOTE_PATH)/output/ ./output

push:
	rsync -rvza -e 'ssh -p 503' . $(HOST):$(REMOTE_PATH)


$(PROJECT).tar: $(SOURCES) $(SPECIAL)
	tar -cf $@ $(SOURCES) $(SPECIAL)

tar: $(PROJECT).tar

