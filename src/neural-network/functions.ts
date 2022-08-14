import { v4 as uuid } from "uuid";
import { Input, Link, Neuron } from "./neural-network";

const initFactor = 1 / 768;

export function createLayers() {
  const inputs = Array(768)
    .fill(null)
    .map(() => new Input(uuid()));
  const layers: (Neuron | Input)[][] = [16, 16, 10].map((num) =>
    Array(num)
      .fill(null)
      .map(() => new Neuron(uuid(), Math.random() * initFactor))
  );
  (layers[0] as Neuron[]).forEach((n) => {
    delete n.expectedOutput;
    delete n.cost;
  });
  (layers[1] as Neuron[]).forEach((n) => {
    delete n.expectedOutput;
    delete n.cost;
  });
  layers.unshift(inputs);
  return layers;
}

export function addLinks(layers: (Neuron | Input)[][]) {
  layers.forEach((layerLeft, i) => {
    if (i === layers.length - 1) {
      layers[i].forEach((output) => {
        delete (output as Neuron).links.right;
      });
      return;
    }
    const layerRight = layers[i + 1];
    layerLeft.forEach((neuronLeft) => {
      layerRight.forEach((neuronRight) => {
        const link = new Link(
          uuid(),
          Math.random() * initFactor,
          neuronLeft,
          neuronRight as Neuron
        );
        neuronLeft.links.right!.push(link);
        (neuronRight as Neuron).links.left.push(link);
      });
    });
  });
  return layers;
}
