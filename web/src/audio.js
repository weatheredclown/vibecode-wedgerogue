/**
 * Placeholder FM-synthesis sound system.
 * The Python version synthesises audio on the fly via NumPy + pygame.mixer.
 * Implementing a faithful port in the browser would require WebAudio work,
 * so for now we keep the interface and leave TODO hooks.
 */
export class SfxSystem {
  constructor() {
    this.enabled = false; // Flip to true once the WebAudio backend is implemented.
  }

  /**
   * @param {number} _index
   */
  play(_index) {
    if (!this.enabled) return;
    // TODO: map preset indices to WebAudio buffers.
  }
}
