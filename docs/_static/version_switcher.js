"use strict";

// Renders a version dropdown in the furo sidebar. The list of published
// versions is read from /versions.json at the site root, which is regenerated
// by scripts/support/update_gh_pages.sh on every docs deployment. On local
// builds or PR previews the fetch fails and the dropdown is simply not shown.
(function () {
  // Matches e.g. "/docs/latest/start/install.html" ->
  //   docsBase="/docs/", currentVersion="latest", pagePath="/start/install.html"
  const match = window.location.pathname.match(/^(.*\/docs\/)([^/]+)(\/.*)?$/);
  if (match === null) {
    return;
  }
  const docsBase = match[1];
  const currentVersion = match[2];
  const pagePath = match[3] || "/";

  // "../versions.json" relative to the docs base resolves to the site root.
  fetch(docsBase + "../versions.json")
    .then((response) => (response.ok ? response.json() : null))
    .then((versions) => {
      if (!Array.isArray(versions) || versions.length === 0) {
        return;
      }

      const select = document.createElement("select");
      select.setAttribute("aria-label", "Documentation version");
      for (const entry of versions) {
        const option = document.createElement("option");
        option.value = entry.version;
        option.textContent = entry.name;
        option.selected = entry.version === currentVersion;
        select.appendChild(option);
      }
      // The currently viewed version may not be in versions.json (e.g. a
      // just-published version before the list refreshes); show it anyway.
      if (select.selectedIndex === -1) {
        const option = document.createElement("option");
        option.value = currentVersion;
        option.textContent = currentVersion;
        option.selected = true;
        select.insertBefore(option, select.firstChild);
      }
      select.addEventListener("change", () => {
        // Keep the current page path; if it does not exist in the target
        // version, the site-wide 404 page falls back to that version's index.
        window.location.href = docsBase + select.value + pagePath;
      });

      const container = document.createElement("div");
      container.className = "sidebar-version-switcher";
      const label = document.createElement("span");
      label.textContent = "Version: ";
      container.appendChild(label);
      container.appendChild(select);

      const style = document.createElement("style");
      style.textContent = [
        ".sidebar-version-switcher {",
        "  margin: 0.5rem 1rem;",
        "  font-size: var(--sidebar-item-font-size, 0.875rem);",
        "  color: var(--color-sidebar-caption-text);",
        "}",
        ".sidebar-version-switcher select {",
        "  background: var(--color-sidebar-background);",
        "  color: var(--color-sidebar-caption-text);",
        "  border: 1px solid var(--color-sidebar-search-border);",
        "  border-radius: 0.25rem;",
        "  padding: 0.125rem 0.25rem;",
        "}",
      ].join("\n");
      document.head.appendChild(style);

      const search = document.querySelector(".sidebar-search-container");
      if (search !== null) {
        search.parentNode.insertBefore(container, search.nextSibling);
        return;
      }
      const brand = document.querySelector(".sidebar-brand");
      if (brand !== null) {
        brand.parentNode.insertBefore(container, brand.nextSibling);
      }
    })
    .catch(() => {});
})();
