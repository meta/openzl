// Copyright (c) Meta Platforms, Inc. and affiliates.

/* eslint-disable @typescript-eslint/no-require-imports */

'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {run} = require('../native-worker.cjs');

const SAMPLE_SIZE = 5_000_000;

function sampleData() {
  const sample = Buffer.alloc(SAMPLE_SIZE);
  const line = Buffer.from('OpenZL compression playground sample data\n');
  for (let offset = 0; offset < sample.length; offset += line.length) {
    line.copy(sample, offset, 0, Math.min(line.length, sample.length - offset));
  }
  return sample;
}

/** Like run(), but resolves with the results, and a job that could not be measured throws, saying why. */
async function runAll(payload) {
  const results = [];
  const failures = [];
  await run(payload, (message) => {
    if (message.type === 'result') {
      results.push(message.result);
    } else if (message.type === 'failure') {
      failures.push(message.failure);
    }
  });
  if (failures.length > 0) {
    throw new Error(failures.map((failure) => failure.message).join('\n'));
  }
  return results;
}

function openZLRow(profile, training) {
  return {rowId: 1, compressor: 'OpenZL', levels: [1], profile, training};
}

async function main() {
  const executable = process.platform === 'win32' ? 'zli.exe' : 'zli';
  const zliPath = path.resolve(__dirname, '..', 'native', 'build', 'Release', executable);
  const temporaryDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'openzl-playground-'));
  try {
    // A space in the path, which zli must be passed as one argument.
    const inputPath = path.join(temporaryDirectory, 'sample input.bin');
    fs.writeFileSync(inputPath, sampleData());

    const results = await runAll({
      zliPath,
      config: {
        input: {path: inputPath},
        iterations: 1,
        compressors: [
          openZLRow('serial', {paretoFrontier: true, candidates: 10}),
          {rowId: 2, compressor: 'zstd', levels: [5]},
          {rowId: 3, compressor: 'gzip', levels: [5]},
        ],
      },
    });
    const openZLResults = results.filter((result) => result.job.rowId === 1);
    if (
      openZLResults.length < 1 ||
      openZLResults.length > 10 ||
      openZLResults.some((result, index) => result.candidate?.index !== index + 1) ||
      results.length !== openZLResults.length + 2 ||
      results.some((result) => result.srcSize !== SAMPLE_SIZE)
    ) {
      throw new Error('Native training returned an invalid result');
    }

    const untrained = await runAll({
      zliPath,
      config: {input: {path: inputPath}, iterations: 1, compressors: [openZLRow('serial', null)]},
    });
    if (untrained.length !== 1 || untrained[0].candidate !== null || untrained[0].srcSize !== SAMPLE_SIZE) {
      throw new Error('Native untrained benchmark returned an invalid result');
    }

    let zliFailure;
    try {
      await runAll({
        zliPath,
        config: {input: {path: inputPath}, iterations: 1, compressors: [openZLRow('not-a-profile', null)]},
      });
    } catch (error) {
      zliFailure = error;
    }
    if (!(zliFailure instanceof Error) || !zliFailure.message.includes("Profile not found: 'not-a-profile'")) {
      throw new Error('zli failure was not returned to the coordinator');
    }

    process.stdout.write(
      `${JSON.stringify({ok: true, openZLCandidates: openZLResults.length, results: results.length, srcSize: results[0].srcSize})}\n`,
    );
  } finally {
    fs.rmSync(temporaryDirectory, {recursive: true, force: true});
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
