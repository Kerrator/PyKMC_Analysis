// Hotkeys for the event viewer.
//
// Arrow keys jump between event frames:
//   ←  initial
//   ↑  saddle  (Down arrow does the same — convenient on Mac trackpads)
//   →  final
//
// We deliberately ignore key events whose target is an input-like widget so
// that typing in the dropdowns and adjusting the slider with arrow keys
// continue to work normally.

(function () {
  function isInteractiveTarget(t) {
    if (!t) return false;
    const tag = t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    if (t.isContentEditable) return true;
    const role = t.getAttribute && t.getAttribute("role");
    if (role === "slider" || role === "combobox" || role === "listbox" ||
        role === "searchbox" || role === "spinbutton") return true;
    // The number input next to the Dash slider also has type=number.
    if (tag === "INPUT" && t.type === "number") return true;
    return false;
  }

  function frameForKey(key) {
    switch (key) {
      case "ArrowLeft":  return "initial";
      case "ArrowUp":
      case "ArrowDown":  return "saddle";
      case "ArrowRight": return "final";
      default:           return null;
    }
  }

  document.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (isInteractiveTarget(e.target)) return;

    const frame = frameForKey(e.key);
    if (!frame) return;

    const input = document.querySelector(
      "#frame-radio input[value='" + frame + "']"
    );
    if (!input) return;

    e.preventDefault();
    if (!input.checked) input.click();
  });
})();
