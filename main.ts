declare global {
    interface Window { webkitAudioContext: typeof AudioContext }
}

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import * as ort from 'onnxruntime-web';
import { setFeatureExtractor, runFeatureExtractor, setClassifier, runClassifier, preprocess, postprocess } from 'web-audio-classifier';

const canvas = document.createElement('canvas');
const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

let executionMode:string;

if (gl && gl instanceof WebGLRenderingContext) {
    console.log('WebGL is supported.');
} else {
    console.log('WebGL is not supported.');
}

if ('gpu' in navigator) {
    console.log('WebGPU is supported in this environment');
} else {
    console.log('WebGPU is not supported in this environment');
}

async function initializeTensorFlow(): Promise<void> {
    await tf.ready();
}

async function processAudio(): Promise<void> {
    const processStartTime = performance.now();
    // const vggishModelURL = './audioset-vggish-3.onnx'
    const vggishModelURL = './vggish-opset-20.onnx'

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    await initializeTensorFlow();

    const inputElement = document.getElementById('audioFileInput') as HTMLInputElement;
    if (!inputElement.files) {
        throw new Error('No files selected');
    }
    const selectedFile = inputElement.files[0];

    const audioBuffer = await readWavFile(selectedFile, audioCtx);

    let pprocOutput;
    if (executionMode == 'onnx') {
        console.log("Running feature extractor with onnx")
        setFeatureExtractor(vggishModelURL);
        pprocOutput = await runFeatureExtractor(audioBuffer);
    }
    else if (executionMode == 'tf-webgl') {
        await tf.setBackend('webgl');
        console.log("Running feature extractor in TJFS with WebGL backend")
        console.log("Loading model")

        let modelLoadStart = performance.now();
        const model = await tf.loadGraphModel('tfjs/model.json');
        let modelLoadEnd = performance.now();
        console.log(`Time taken for model load: ${modelLoadEnd - modelLoadStart} milliseconds`);

        pprocOutput = await runInferenceTF(audioBuffer, model)
    }
    else if (executionMode == 'tf-webgpu') {
        await tf.setBackend('webgpu');
        console.log("Running feature extractor in TJFS with WebGPU backend")
        console.log("Loading model")

        let modelLoadStart = performance.now();
        const model = await tf.loadGraphModel('tfjs/model.json');
        let modelLoadEnd = performance.now();
        console.log(`Time taken for model load: ${modelLoadEnd - modelLoadStart} milliseconds`);

        pprocOutput = await runInferenceTF(audioBuffer, model)
    }
    if (!pprocOutput) {
        throw new Error('pprocOutput is null or undefined');
    }
    const processEndTime = performance.now();
    console.log(`Absolute total time for feature extraction: ${processEndTime - processStartTime} milliseconds`);

    console.log("post-processed feature extraction output: ")
    console.log(pprocOutput);

    // Load bg noise detector model
    // const bgModelPath = './bg_noise_detection.onnx';
    // setClassifier(bgModelPath)
    // runClassifier(pprocOutput);
    // const processEndTime = performance.now();

    // console.log(`Time taken for overall processing: ${processEndTime - processStartTime} milliseconds`);
        
}

async function runInferenceTF(audioBuffer: AudioBuffer, model: tf.GraphModel) {
    let preprocessStart = performance.now();
    const inputData = await preprocess(audioBuffer);
    let preprocessEnd = performance.now();

    console.log(`Time taken for preprocess: ${preprocessEnd - preprocessStart} milliseconds`);

    try {
        const ortOutputsList = [];

        if (!inputData) {
            throw new Error('inputData is undefined');
        }

        const [batchSize, channels, height, width] = inputData.shape;

        let inferenceStart = performance.now();
        for (let batch = 0; batch < batchSize; batch++) {
            // Assuming input_data is a 3D tensor (similar to permute(2, 1, 0) in Python)
            const input_data_batch = inputData.slice([batch, 0, 0, 0], [1, 1, height, width]);

            const input_data_tfjs = input_data_batch.transpose([0, 2, 1, 3]).reshape([1, 64, 96]);

            // Run inference using the TensorFlow.js model
            const outputTensor = model.execute(input_data_tfjs);

            // Convert the output tensor to a flat array
            let outputArray;
            if (outputTensor instanceof tf.Tensor) {
                outputArray = Array.from(await outputTensor.data());
            } else if (Array.isArray(outputTensor) && outputTensor.length > 0) {
                outputArray = Array.from(await outputTensor[0].data());
            } else {
                throw new Error('Unexpected output from model execution');
            }

            // Push the output to the list
            ortOutputsList.push(outputArray);
        }
        let inferenceEnd = performance.now();

        console.log(`Time taken for VGGish inference (hot): ${inferenceEnd - inferenceStart} milliseconds`);

        console.log("ortOutputsList");
        console.log(ortOutputsList);

        let postProcessStart = performance.now();
        const output = ortOutputsList ? postprocess(ortOutputsList) : null;
        let postProcessEnd = performance.now();

        console.log(`Time taken for postprocessing: ${postProcessEnd - postProcessStart} milliseconds`);
        return output;
    } catch (error) {
        console.error('Error during inference:', error);
        return null;
    }
}

const processAudioButton = document.getElementById('processAudioButton');
const processWASMButton = document.getElementById('processAudioWebGL');
const processWebGPUButton = document.getElementById('processAudioWebGPU');

if (processAudioButton) {
    processAudioButton.addEventListener('click', processOnnx);
}
if (processWASMButton) {
    processWASMButton.addEventListener('click', processTFWebGL);
}
if (processWebGPUButton) {
    processWebGPUButton.addEventListener('click', processTFWebGPU);
}

function processOnnx() {
    executionMode = 'onnx'
    processAudio()
}

function processTFWebGL() {
    executionMode = 'tf-webgl'
    processAudio()
}

function processTFWebGPU() {
    executionMode = 'tf-webgpu'
    processAudio()
}


function readWavFile(file: File, audioCtx: AudioContext): Promise<AudioBuffer> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            if (e.target) {
                audioCtx.decodeAudioData(e.target.result as ArrayBuffer, (buffer) => {
                    resolve(buffer);
                });
            } else {
                reject(new Error('FileReader event target is null'));
            }
        };
        reader.onerror = function (e) {
            reject(e);
        };
        reader.readAsArrayBuffer(file);
    });
}
