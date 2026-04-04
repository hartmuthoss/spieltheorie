rem set PATH=f:\Programme\GraphViz-2.38\bin;%PATH%
set PATH=f:\Programme\Graphviz-14.1.1\bin;%PATH%
pushd ..\docs
rem for %%f in ( bayesian_equilibrium_example2_1 bayesian_equilibrium_example2_2 bayesian_equilibrium_example2_3 bier_quiche pbe_two_gas_stations hilbert_gas_stations_equilibrium1 prisoners_dilemma_repeated_05b_markov ) do (
for %%f in ( prisoners_dilemma_repeated_05b_markov ) do (
  del /f/q %%f.svg
  dot -Tsvg -o %%f.svg %%f.dot
  start %%f.svg
)
popd
@pause
