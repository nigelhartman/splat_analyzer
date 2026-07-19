// Lower-left FAB with a popover holding stats, an examples dropdown, and file pickers.
// Pure UI: emits example ids and chosen File objects via callbacks.
const GEAR = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`;

export class SettingsMenu {
  constructor({ examples, defaultExampleId, onExampleSelect, onSplatFile, onAnnotationsFile }) {
    const exampleOptions = examples
      .map(
        (ex) =>
          `<option value="${ex.id}"${ex.id === defaultExampleId ? " selected" : ""}>${ex.label}</option>`
      )
      .join("");

    const root = document.createElement("div");
    root.id = "settings";
    root.innerHTML = `
      <div class="menu-panels">
        <div class="menu">
          <h2>Statistics</h2>
          <div class="stat-line" data-stat="fps">FPS —</div>
          <div class="stat-line" data-stat="splats">— mio Splats</div>
          <div class="stat-line" data-stat="labels">0 Labels</div>
        </div>
        <div class="menu">
          <h2>Load scene</h2>
          <div class="field">
            <label>Examples</label>
            <select data-input="example">
              <option value="">None</option>
              ${exampleOptions}
            </select>
          </div>
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
      </div>
      <button class="fab" title="Settings">${GEAR}</button>`;
    document.body.appendChild(root);

    this._root = root;
    this._panels = root.querySelector(".menu-panels");
    this._exampleSelect = root.querySelector('[data-input="example"]');
    this._statFps = root.querySelector('[data-stat="fps"]');
    this._statSplats = root.querySelector('[data-stat="splats"]');
    this._statLabels = root.querySelector('[data-stat="labels"]');
    this._open = false;
    this._lastStatsKey = "";
    this._pendingStats = { fps: 0, splatCount: 0, labelCount: 0 };

    const fab = root.querySelector(".fab");
    fab.addEventListener("click", (e) => {
      e.stopPropagation();
      this._setOpen(!this._open);
    });
    document.addEventListener("click", (e) => {
      if (!root.contains(e.target)) this._setOpen(false);
    });

    this._exampleSelect.addEventListener("change", () => {
      onExampleSelect(this._exampleSelect.value);
    });

    this._wireInput(root, "splat", "Choose .ply…", onSplatFile);
    this._wireInput(root, "json", "Choose .json…", onAnnotationsFile);
  }

  updateStats(stats) {
    this._pendingStats = stats;
    if (!this._open) return;
    this._renderStats();
  }

  clearFileSelections() {
    this._resetFileField("splat", "Choose .ply…");
    this._resetFileField("json", "Choose .json…");
  }

  setExampleId(id) {
    this._exampleSelect.value = id;
  }

  _setOpen(open) {
    this._open = open;
    this._panels.classList.toggle("open", open);
    this._root.querySelector(".fab").classList.toggle("open", open);
    if (open) {
      this._lastStatsKey = "";
      this._renderStats();
    }
  }

  _renderStats() {
    const { fps, splatCount, labelCount } = this._pendingStats;
    const fpsText = Number.isFinite(fps) ? Math.round(fps) : "—";
    const splatsText =
      splatCount > 0 ? `${(splatCount / 1_000_000).toFixed(2)} mio Splats` : "— mio Splats";
    const labelsText = `${labelCount || 0} Labels`;
    const key = `${fpsText}|${splatsText}|${labelsText}`;
    if (key === this._lastStatsKey) return;
    this._lastStatsKey = key;
    this._statFps.textContent = `FPS ${fpsText}`;
    this._statSplats.textContent = splatsText;
    this._statLabels.textContent = labelsText;
  }

  _resetFileField(key, placeholder) {
    const btn = this._root.querySelector(`[data-target="${key}"]`);
    const input = this._root.querySelector(`[data-input="${key}"]`);
    btn.textContent = placeholder;
    btn.classList.remove("loaded");
    input.value = "";
  }

  _wireInput(root, key, placeholder, callback) {
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
