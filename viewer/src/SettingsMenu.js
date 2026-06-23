// Lower-left FAB with a popover holding two file pickers. Pure UI: it emits the
// chosen File objects via callbacks and knows nothing about the scene.
const GEAR = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`;

export class SettingsMenu {
  constructor({ onSplatFile, onAnnotationsFile }) {
    const root = document.createElement("div");
    root.id = "settings";
    root.innerHTML = `
      <div class="menu">
        <h2>Load scene</h2>
        <div class="field">
          <label>Gaussian splat</label>
          <button class="file-btn" data-target="splat">Choose .ply…</button>
          <input type="file" accept=".ply,.spz,.splat,.ksplat,.rad" data-input="splat">
        </div>
        <div class="field">
          <label>Annotations</label>
          <button class="file-btn" data-target="json">Choose .json…</button>
          <input type="file" accept=".json,application/json" data-input="json">
        </div>
      </div>
      <button class="fab" title="Settings">${GEAR}</button>`;
    document.body.appendChild(root);

    const fab = root.querySelector(".fab");
    const menu = root.querySelector(".menu");
    fab.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = menu.classList.toggle("open");
      fab.classList.toggle("open", open);
    });
    document.addEventListener("click", (e) => {
      if (!root.contains(e.target)) {
        menu.classList.remove("open");
        fab.classList.remove("open");
      }
    });

    this._wireInput(root, "splat", onSplatFile);
    this._wireInput(root, "json", onAnnotationsFile);
  }

  _wireInput(root, key, callback) {
    const btn = root.querySelector(`[data-target="${key}"]`);
    const input = root.querySelector(`[data-input="${key}"]`);
    btn.addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
      const file = input.files[0];
      if (!file) return;
      btn.textContent = file.name;
      btn.classList.add("loaded");
      callback(file);
    });
  }
}
