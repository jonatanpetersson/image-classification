import logo from './logo.svg';
import './App.css';
import { Network } from './neural-network/neural-network';
import {
  imageData,
  learningRate,
  networkData,
} from './neural-network/constants';

function App() {
  const network = new Network(networkData, learningRate, imageData);
  let iteration = network.iteration;

  function handleIteration() {
    network.iterate();
  }

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
        <p>Iteration: {iteration}</p>
        <button onClick={handleIteration}>Iterate</button>
      </header>
    </div>
  );
}

export default App;
