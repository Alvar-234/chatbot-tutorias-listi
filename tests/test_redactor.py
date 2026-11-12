from chatbot.privacy.redactor import Redactor

r = Redactor()


def test_correo_y_telefono():
    out = r.redact("Mi correo es ana@uv.mx y mi tel 228 123 4567")
    assert "ana@uv.mx" not in out.text and "4567" not in out.text
    assert set(out.types) == {"correo", "telefono"}


def test_curp_y_matricula():
    out = r.redact("CURP GOMA990101HVZRRN09 matrícula zS20012345")
    assert "[CURP]" in out.text and "[MATRICULA]" in out.text


def test_no_toca_cifras_normales():
    txt = "¿Cuántos créditos hay en 2026 y en 8 periodos?"
    assert r.redact(txt).text == txt
