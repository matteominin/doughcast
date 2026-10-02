# Static UI manual checklist

The page is vanilla HTML/CSS/JS and has no runtime CDN dependency. Chart.js was not available as a local asset in this workspace, so the history view uses a small canvas renderer with the same actual/predicted data contract. The page shell still loads without the API; API-backed flows need the local FastAPI server.

## Browser checks

- [ ] Open at `http://127.0.0.1:<port>/` and confirm the shell loads.
- [ ] Check 390 x 844: no horizontal scroll, bottom tabs remain reachable, and controls do not overlap.
- [ ] Check desktop: the tab bar becomes compact top navigation and content remains centered.
- [ ] Use keyboard Tab navigation: focus is visible, controls have Italian accessible names, and the main view receives focus after tab changes.
- [ ] On localhost, allow the microphone. Tap `Registra`, confirm the timer changes, tap again, and confirm transcript plus editable fields appear.
- [ ] Deny microphone permission or use an insecure non-local origin. Confirm the Italian limitation message appears and manual entry remains available.
- [ ] Edit extracted values, leave optional values blank, and confirm `Rifai` does not save anything.
- [ ] Submit a manual record. Confirm `POST /api/records` happens only after `Salva` and API errors are visible.
- [ ] With fewer than required rows, confirm `Domani` shows progress and no forecast number.
- [ ] With a ready response, confirm balls, grams, p10/p50/p90, model, training days, and service-level trade-offs.
- [ ] Open `Storico`: confirm recent records, `a memoria` badges, empty/loading/error states, and chart lines.
- [ ] Click `Modifica`, change a value, and save. Confirm the date is preserved for the upsert.
- [ ] Disconnect the network and reload. Confirm no external asset request is needed for the shell or chart.