"""Comprobación de la aplicación Streamlit pública con Playwright.

Busca el dashboard tanto en la página principal como en marcos embebidos.
Genera imágenes de diagnóstico si algún control no se puede automatizar.
"""
import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://casino-estadistico-ubpy8lsfkqbb8yh6teansn.streamlit.app/")
OUT = Path("artifacts/live-smoke")
OUT.mkdir(parents=True, exist_ok=True)


def find_application(page, timeout=120):
    deadline = time.monotonic() + timeout
    last_report = 0
    while time.monotonic() < deadline:
        for frame in page.frames:
            try:
                text = frame.locator("body").inner_text(timeout=1800)
                if "Casino" in text and "Estadístico" in text and "Lanzar" in text:
                    print(f"Found Streamlit UI in frame: {frame.url}", flush=True)
                    return frame
            except Exception:
                continue
        if time.monotonic() - last_report > 20:
            last_report = time.monotonic()
            print(f"Waiting for UI; frames={len(page.frames)}", flush=True)
            for frame in page.frames[:5]:
                try:
                    preview = frame.locator("body").inner_text(timeout=1000)[:240].replace("\n", " ")
                    print(f"Frame {frame.url}: {preview!r}", flush=True)
                except Exception as exc:
                    print(f"Frame inaccessible ({str(exc)[:120]})", flush=True)
        page.wait_for_timeout(2000)
    raise AssertionError("UI not found in root document or child frames.")


def check_desktop(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 960})
    try:
        response = page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        print(f"Desktop HTTP {response.status if response else 'unknown'}", flush=True)
        app = find_application(page)
        page.screenshot(path=str(OUT / "desktop-inicio.png"), full_page=True)
        assert app.get_by_text("Mesa estadística", exact=False).count() >= 1
        assert app.get_by_text("Calibración", exact=True).count() >= 1
        print("OK: dashboard title, controls and calibration label", flush=True)

        app.get_by_role("button", name="▶ Lanzar 100 tiradas").click(timeout=18000)
        app.get_by_text("Tiradas acumuladas").wait_for(timeout=18000)
        print("OK: manual simulation button clicked", flush=True)

        # The tab label may be inside a button, depending on Streamlit version.
        tab = app.get_by_role("tab", name="Calibración")
        if tab.count():
            tab.click(timeout=15000)
        else:
            app.get_by_text("Calibración", exact=True).first.click(timeout=15000)
        app.get_by_text("Calibración: observado vs. esperado").wait_for(timeout=18000)
        page.screenshot(path=str(OUT / "desktop-calibracion.png"), full_page=True)
        print("OK: calibration tab displayed", flush=True)
    except Exception:
        page.screenshot(path=str(OUT / "desktop-error.png"), full_page=True)
        raise
    finally:
        page.close()


def check_mobile(browser):
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
    try:
        response = page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        print(f"Mobile HTTP {response.status if response else 'unknown'}", flush=True)
        frame = find_application(page, timeout=90)
        page.screenshot(path=str(OUT / "mobile-inicio.png"), full_page=True)
        width = frame.evaluate("document.documentElement.scrollWidth")
        print(f"Mobile width: {width}px (viewport 390px)", flush=True)
        if width > 430:
            raise AssertionError(f"Mobile horizontal overflow: {width}px")
        print("OK: mobile rendered without major overflow", flush=True)
    except Exception:
        page.screenshot(path=str(OUT / "mobile-error.png"), full_page=True)
        raise
    finally:
        page.close()


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            check_desktop(browser)
            check_mobile(browser)
        finally:
            browser.close()
    print("LIVE SMOKE PASSED", flush=True)


if __name__ == "__main__":
    main()
