import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from court_slicer import MPVController, is_wayland_session


class WaylandSessionTests(unittest.TestCase):
    def test_deve_detectar_wayland_pelo_tipo_da_sessao(self) -> None:
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True):
            self.assertTrue(is_wayland_session())

    def test_deve_detectar_wayland_pela_variavel_de_display(self) -> None:
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True):
            self.assertTrue(is_wayland_session())

    def test_nao_deve_detectar_wayland_em_sessao_x11(self) -> None:
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "x11"}, clear=True):
            self.assertFalse(is_wayland_session())


class EmbeddedVideoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = MPVController(Path("video.mp4"))

    def test_deve_embutir_video_no_linux_com_x11(self) -> None:
        environment = {"XDG_SESSION_TYPE": "x11"}
        with patch.object(sys, "platform", "linux"), patch.dict(
            os.environ, environment, clear=True
        ):
            self.assertTrue(self.controller.uses_embedded_video)

    def test_deve_abrir_janela_separada_no_linux_com_wayland(self) -> None:
        environment = {
            "XDG_SESSION_TYPE": "wayland",
            "WAYLAND_DISPLAY": "wayland-0",
        }
        with patch.object(sys, "platform", "linux"), patch.dict(
            os.environ, environment, clear=True
        ):
            self.assertFalse(self.controller.uses_embedded_video)

    def test_deve_abrir_janela_separada_no_windows(self) -> None:
        with patch.object(sys, "platform", "win32"), patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertFalse(self.controller.uses_embedded_video)

    def test_deve_abrir_janela_separada_no_macos(self) -> None:
        with patch.object(sys, "platform", "darwin"), patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertFalse(self.controller.uses_embedded_video)

    def test_nao_deve_enviar_wid_ao_mpv_no_wayland(self) -> None:
        environment = {"XDG_SESSION_TYPE": "wayland"}
        with patch.object(sys, "platform", "linux"), patch.dict(
            os.environ, environment, clear=True
        ), patch.object(self.controller, "_ipc_ready", return_value=True), patch(
            "court_slicer.subprocess.Popen"
        ) as popen:
            self.controller.start(wid=12345)

        command = popen.call_args.args[0]
        self.assertFalse(any(argument.startswith("--wid=") for argument in command))
        self.assertIn("--vo=wlshm", command)
        self.assertIn("--hwdec=no", command)
        self.assertEqual(subprocess.DEVNULL, popen.call_args.kwargs["stderr"])

    def test_deve_enviar_wid_ao_mpv_no_x11(self) -> None:
        environment = {"XDG_SESSION_TYPE": "x11"}
        with patch.object(sys, "platform", "linux"), patch.dict(
            os.environ, environment, clear=True
        ), patch.object(self.controller, "_ipc_ready", return_value=True), patch(
            "court_slicer.subprocess.Popen"
        ) as popen:
            self.controller.start(wid=12345)

        command = popen.call_args.args[0]
        self.assertIn("--wid=12345", command)
        self.assertNotIn("--vo=wlshm", command)
        self.assertNotIn("--hwdec=no", command)


if __name__ == "__main__":
    unittest.main()
