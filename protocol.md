# Experiment specification

Question: how do constant input and absolute refractory time change a leaky integrate-and-fire neuron's firing?

All data are simulated. Fixed values: resting/reset voltage −65 mV, threshold −50 mV, time constant 20 ms, resistance 100 MΩ. Initial voltage equals reset. Sweep current from 0 to 0.5 nA in 0.025 nA increments and recovery times 1, 2 and 5 ms. Run for 1,000 ms with 0.1 ms Euler steps; count events in (100, 1,000] ms. The analytic rate refers to the steady continuous model, whereas the simulation rate is a finite-window count.

Events occur at the right edge of a step, immediately reset the voltage, and start an integer number of full refractory steps. No artificial spike waveform is drawn. Repeat with 0.05 ms steps; allowed rate difference is the larger of 2 Hz or 2% of the refined rate. Independently compare subthreshold voltage and firing rate with analytic solutions. At exactly 0.15 nA, the continuous voltage approaches threshold asymptotically without firing.

These are teaching parameters, not a fit to a biological cell. Deterministic repetitions are not biological replicates. No external measurements or statistical generalisation are involved. The model omits ion-channel dynamics, adaptation, morphology and synaptic noise.
