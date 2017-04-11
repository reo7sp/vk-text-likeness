FROM python:3-onbuild
MAINTAINER Oleg Morozenkov <a@reo7sp.ru>

ENTRYPOINT [ "python", "./main.py" ]