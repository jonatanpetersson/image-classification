import { Network, Neuron } from './neural-network';

const totalCost = (network: Network) =>
  network.outputLayer.reduce((sum, n: Neuron) => sum + n.cost!, 0);

test('has one input per pixel', () => {
  const network = new Network();
  expect(network.layers[0]).toHaveLength(network.batch[0].pixels.length);
});

test('training on one sample lowers its cost', () => {
  const network = new Network();
  network.initializeSample(0);
  network.propagateForward();
  const before = totalCost(network);
  for (let i = 0; i < 20; i++) {
    network.propagateForward();
    network.propagateBack();
  }
  network.propagateForward();
  expect(totalCost(network)).toBeLessThan(before / 2);
  expect(network.guessedDigit()).toBe(network.batch[0].label);
});

test('accuracy improves well beyond chance after a few epochs', () => {
  const network = new Network();
  let accuracy = 0;
  for (let epoch = 0; epoch < 5; epoch++) accuracy = network.runEpoch();
  // Random guessing gives about 10%.
  expect(accuracy).toBeGreaterThan(0.6);
});
