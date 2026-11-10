import './App.css';
import { Network } from './neural-network/neural-network';
import { useState } from 'react';

export const network = new Network();

function App() {
  const [outputUI, setOutputUI] = useState(network.outputLayer.map(o => o));
  const [sample, setSample] = useState(0);
  const [passes, setPasses] = useState(1);
  const [accuracy, setAccuracy] = useState<number[]>([]);
  const handleSampleInput = (event: React.ChangeEvent<HTMLInputElement>) => setSample(parseInt(event.target.value) || 0);
  const handlePassesInput = (event: React.ChangeEvent<HTMLInputElement>) => setPasses(parseInt(event.target.value) || 0);
  const initializeSample = () => network.initializeSample(sample);
  const runEpoch = () => {
    console.time('Running epoch');
    const result = network.runEpoch();
    console.timeEnd('Running epoch');
    setAccuracy(prev => [...prev, result]);
  };

  const guess = () => {
    network.propagateForward();
    setOutputUI(network.outputLayer.map(o => o));
  };
  const train = () => {
    for (let i = 0; i < passes; i++) {
      network.propagateForward();
      network.propagateBack();
    }
    network.propagateForward();
    setOutputUI(network.outputLayer.map(o => o));
  };
  return (
    <div className='image-classification'>
      <h1>Image classification</h1>
      <h3>Output | Cost | Expected output</h3>
      <ul>
        {outputUI.map((output, i) =>
          <li key={i} >{i} --- {output.activation.toFixed(10)} | {output.cost?.toFixed(10)} | {output.expectedOutput}</li>
        )}
      </ul>
      <br />
      <canvas></canvas>
      <button onClick={runEpoch}>Run epoch ({network.batch.length} samples)</button>
      {accuracy.length > 0 && (
        <p>Accuracy per epoch: {accuracy.map(a => `${(a * 100).toFixed(1)}%`).join(', ')}</p>
      )}
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
