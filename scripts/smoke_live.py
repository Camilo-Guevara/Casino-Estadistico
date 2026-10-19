"""Prueba de humo del Casino Estadístico desplegado en Streamlit Cloud.

Verifica renderizado real con Chromium, interacción básica, calibración y
capturas de escritorio/móvil. No sustituye una revisión humana de estética.
"""
import os
import re
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

URL = os.environ.get(
    "APP_URL",
    "https://casino-estadistico-ubpy8lsfkqbb8yh6teansn.streamlit.app/",
)
ARTIFACTS = Path("artifacts/live-smoke")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def check_desktop(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1)
    page.on("pageerror", lambda error: print(f"Browser JavaScript error: {error}"))
    try:
        deadline = time.monotonic() + 180
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            try:
                response = page.goto(URL, wait_until="domcontentloaded", timeout=45000)
                if response is not None:
                    print(f"HTTP status on attempt {attempts}: {response.status}")
                page.get_by_role("heading", name=re.compile("Casino\\s+Estadístico", re.I)).wait_for(timeout=23000)
                break
            except (PlaywrightTimeout, Exception) as exc:
                print(f"Waiting for Streamlit to start: {type(exc).__name__}: {str(exc)[:160]}")
                page.wait_for_timeout(2500)
        else:
            raise AssertionError("Casino Estadístico was not rendered within 180 seconds")

        print("OK: heading Casino Estadístico rendered")
        assert page.get_by_text("Mesa estadística", exact=False).count() >= 1
        assert page.get_by_role("tab", name="Calibración").count() >= 1
        assert page.locator('[data-testid="stPlotlyChart"]').count() >= 1
        page.screenshot(path=str(ARTIFACTS / "desktop-inicio.png"), full_page=True)
        print("OK: controls and Plotly chart present")

        page.get_by_role("button", name=re.compile("Lanzar")).first.click()
        page.get_by_text("Tiradas acumuladas", exact=False).first.wait_for(timeout=30000)
        page.get_by_role("tab", name="Calibración").click()
        page.get_by_text("Calibración: observado vs. esperado").wait_for(timeout=30000)
        assert page.get_by_text("Tasa observada", exact=False).count() >= 1
        page.screenshot(path=str(ARTIFACTS / "desktop-calibracion.png"), full_page=True)
        print("OK: simulated a batch and rendered calibration tab")
        return True
    except Exception:
        page.screenshot(path=str(ARTIFACTS / "desktop-error.png"), full_page=True)
        raise
    finally:
        page.close()


def check_mobile(browser):
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1, is_mobile=True, has_touch=True)
    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=45000)
        page.get_by_role("heading", name=re.compile("Casino\\s+Estadístico", re.I)).wait_for(timeout=90000)
        page.screenshot(path=str(ARTIFACTS / "mobile-inicio.png"), full_page=True)
        width = page.evaluate("document.documentElement.scrollWidth")
        print(f"Mobile screenshot saved (document scroll width: {width}px, viewport: 390px)")
        # Plotly/Streamlit may use a few pixels of rounding; reject major overflow.
        if width > 430:
            raise AssertionError(f"Excessive horizontal mobile overflow: {width}px")
        print("OK: mobile page renders without major horizontal overflow")
    except Exception:
        page.screenshot(path=str(ARTIFACTS / "mobile-error.png"), full_page=True)
        raise
    finally:
        page.close()


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            check_desktop(browser)
            check_mobile(browser)
        finally:
            browser.close()
    print(f"LIVE SMOKE PASSED: {URL}")


if __name__ == "__main__":
    main()
