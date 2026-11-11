# Build the Beamer PDF
PDF = main.pdf
TEX = main.tex

.PHONY: all clean figures

all: $(PDF)

$(PDF): $(TEX) figures
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)
	cp $(PDF) ../Presentation_Beamer.pdf
	cp $(PDF) ../../Presentation_Beamer.pdf

# Re-export SVG charts from ../assets if needed
figures:
	@mkdir -p figures
	@cp -n ../assets/*.png figures/ 2>/dev/null || true
	@for f in c_offline_variants c_stream_s1 c_stream_s2 c_stream_bars \
	          c_ablation_acc c_ablation_f1 c_ks_bars c_runtime \
	          c_frozen_decay c_gap c_per_source i_selection i_ks i_boundary; do \
	  if [ -f ../assets/$$f.svg ] && [ ! -f figures/$$f.pdf ]; then \
	    inkscape ../assets/$$f.svg --export-type=pdf --export-filename=figures/$$f.pdf; \
	  fi; \
	done

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb
