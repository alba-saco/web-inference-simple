declare global {
    interface Window { webkitAudioContext: typeof AudioContext }
}

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { preprocess, postprocess } from 'web-audio-classifier';

async function initializeTensorFlow(): Promise<void> {
    const backend = 'gpu' in navigator ? 'webgpu' : 'webgl';
    await tf.setBackend(backend);
    await tf.ready();
    console.log(`Using TensorFlow backend: ${backend}`);
}

async function processAudio(): Promise<void> {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    console.time('tfjs init')
    await initializeTensorFlow();
    console.timeEnd('tfjs init')

    const inputElement = document.getElementById('audioFileInput') as HTMLInputElement;
    if (!inputElement.files) {
        throw new Error('No files selected');
    }
    const selectedFile = inputElement.files[0];

    const audioBuffer = await readWavFile(selectedFile, audioCtx);

    console.time('model load')
    const model = await tf.loadGraphModel('tfjs/model.json');
    console.timeEnd('model load')

    console.time('total inference')
    const pprocOutput = await runInferenceTF(audioBuffer, model)
    console.timeEnd('total inference')
    if (!pprocOutput) throw new Error('pprocOutput is null or undefined');
}

async function runInferenceTF(audioBuffer: AudioBuffer, model: tf.GraphModel) {
    console.time('preprocess')
    const inputData = await preprocess(audioBuffer);
    console.timeEnd('preprocess')

    try {
        const ortOutputsList = [];

        if (!inputData) {
            throw new Error('inputData is undefined');
        }

        const [batchSize, channels, height, width] = inputData.shape;

        console.time('vggish inference')
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
        console.timeEnd('vggish inference')
        
        console.time('postprocess')
        const output = ortOutputsList ? postprocess(ortOutputsList) : null;
        console.timeEnd('postprocess')
        return output;
    } catch (error) {
        console.error('Error during inference:', error);
        return null;
    }
}

const processAudioButton = document.getElementById('processAudioButton');
processAudioButton?.addEventListener('click', processAudio);

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
