import './App.css';
import { Network, Neuron } from './neural-network/neural-network';
import { useState, createRef } from 'react';

export const network = new Network()
let numCounter = Array(10).fill(0);
for (let i = 0; i < 999; i++) {
  numCounter[network.batch[i].label] += 1;
}
console.log(numCounter)

function App() {
  const ulRef = createRef();
  const [outputUI, setOutputUI] = useState(network.layers[3])
  const [sample, setSample] = useState(0);
  const [passes, setPasses] = useState(1);
  const handleSampleInput = (event: any) => setSample(parseInt((event.target as HTMLInputElement).value));
  const handlePassesInput = (event: any) => setPasses(parseInt((event.target as HTMLInputElement).value));
  const initializeSample = () => network.initializeSample(sample);
  const runEpoch = () => {
    console.time('Running epoch')
    for (let i = 0; i < 99; i++) {
      console.log(i);
      network.initializeSample(i);
      network.propagateForward();
      network.propagateBack();
    }
    console.timeEnd('Running epoch')
  }

  const guess = () => {
    network.propagateForward();
    // network.propagateBack();
    setOutputUI(network.layers[3].map(o => o));
  };
  const train = () => {
    if (passes > 0) {
      for (let i = 0; i < passes; i++) {
        console.log(sample, passes)
        network.propagateForward();
        network.propagateBack();
        setOutputUI(() => network.layers[3].map(o => o));
      }
    } 
    console.log(network.layers[1][0])
  };
  return (
    <div className='image-classification'>
      <h1>Image classification</h1>
      <h3>Output | Cost | Expected output</h3>
      <ul>
        {outputUI.map((output, i) => 
          <li key={i} >{i} --- {output.activation.toFixed(10)} | {(output as Neuron).cost?.toFixed(10)} | {(output as Neuron).expectedOutput}</li>
        )}
      </ul>
      <br />
      <canvas></canvas>
      <button onClick={runEpoch}>Run epoch (1000 samples)</button>
      <p>Set sample</p>
      <input type="number" onChange={handleSampleInput} />
      <p>Set passes per sample</p>
      <input type="number" onChange={handlePassesInput} />
      <button onClick={initializeSample}>Initialize sample</button>
      <button onClick={train}>Train</button>
      <button onClick={guess}>Guess</button>
    </div>
  );
}

export default App;