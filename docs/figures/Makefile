# Схемы архитектур (D2, https://d2lang.com): make -C docs/figures
#   make           — SVG всех схем
#   make png       — ещё и PNG (D2 при первом запуске скачивает Chromium)
#   make watch F=twotower_lazy — живой предпросмотр в браузере, пересборка при сохранении
# Установка: brew install d2  (или curl -fsSL https://d2lang.com/install.sh | sh -s --)

D2 ?= d2
SRC := $(filter-out style.d2 legend.d2,$(wildcard *.d2))
COMMON := style.d2 legend.d2

svg: $(SRC:.d2=.svg)
png: svg $(SRC:.d2=.png)

%.svg: %.d2 $(COMMON)
	$(D2) $< $@

%.png: %.d2 $(COMMON)
	$(D2) $< $@

watch:
	$(D2) --watch $(F).d2 $(F).svg

fmt:
	$(D2) fmt $(SRC) $(COMMON)

clean:
	rm -f $(SRC:.d2=.svg) $(SRC:.d2=.png)

.PHONY: svg png watch fmt clean
