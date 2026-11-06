// Copyright (c) Meta Platforms, Inc. and affiliates.

// `?url` emits the file as an asset of its own rather than into a bundle, so
// the page downloads it only when someone asks for the sample.
import sampleUrl from '../../../../examples/getting_started/sample_inputs/sra0.bin?url';

/** What the sample is called once chosen, as a file the user picked would be. */
export const SAMPLE_NAME = 'sra0.bin';

/**
 * Downloads one of OpenZL's example inputs, to run as if the user had chosen it.
 * Every way the download can fail -- an error status, no network, a body cut
 * off partway -- rejects with a message that says it was the sample.
 */
export async function fetchSample(): Promise<File> {
  let bytes: ArrayBuffer;
  try {
    const response = await fetch(sampleUrl);
    if (!response.ok) {
      // HTTP/2 has no reason phrase, so `statusText` is often empty.
      throw new Error(`${String(response.status)} ${response.statusText}`.trim());
    }
    bytes = await response.arrayBuffer();
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new Error(`Could not download the sample: ${reason}`, {cause: error});
  }
  return new File([bytes], SAMPLE_NAME);
}
