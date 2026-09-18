TECTONIC ?= tectonic

.PHONY: all figures figure3 figure4 clean

all: figures main.pdf

figures: figure3 figure4
	python3 make_figures.py

figure3:
	python3 figure3_bestpaper/build_figure3.py

figure4:
	python3 figure4_lm_validation/build_lm_validation_figure.py

main.pdf: main.tex refs.bib figures/fig_same_lambda.pdf figures/fig_schematic.pdf figures/fig_sae_wall.pdf figures/fig_sae_vs_ind.pdf figures/fig_sae_floor_supplied.pdf figures/fig_gpt2.pdf figures/fig_sae_exponent.pdf figures/fig_public_atoms_supplied.pdf figures/fig1_detector_boundary.png figures/fig2_detector_decoder.pdf figures/figA_residual_floor.png figures/figA_public_atoms.png figure3_bestpaper/output/figure3_spectral_warm_start.pdf figure4_lm_validation/output/figure_lm_validation.pdf
	$(TECTONIC) -X compile --untrusted main.tex

clean:
	rm -f main.pdf main.aux main.log main.bbl main.blg
