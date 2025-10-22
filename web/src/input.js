export class InputManager {
  constructor() {
    this.keys = new Set();
    this.justPressed = new Set();
    this.justReleased = new Set();
    this.listeners = [];
    window.addEventListener("keydown", (e) => this.onKeyDown(e));
    window.addEventListener("keyup", (e) => this.onKeyUp(e));
  }

  onKeyDown(event) {
    if (event.repeat) return;
    this.keys.add(event.code);
    this.justPressed.add(event.code);
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"].includes(event.code)) {
      event.preventDefault();
    }
  }

  onKeyUp(event) {
    this.keys.delete(event.code);
    this.justReleased.add(event.code);
  }

  isDown(code) {
    return this.keys.has(code);
  }

  wasPressed(code) {
    return this.justPressed.has(code);
  }

  wasReleased(code) {
    return this.justReleased.has(code);
  }

  tick() {
    this.justPressed.clear();
    this.justReleased.clear();
  }
}
