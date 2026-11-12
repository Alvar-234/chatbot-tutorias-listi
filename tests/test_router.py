from chatbot.domain.models import Route


def _router(build):
    return build(None).router


def test_riesgo_deriva(build):
    d = _router(build).route("a veces pienso en el suicidio")
    assert d.route == Route.HANDOFF and d.handoff_kind == "risk"


def test_tramite_deriva(build):
    assert _router(build).route("inscríbeme a bases de datos").handoff_kind == "administrative"


def test_seriacion_estructurada(build):
    d = _router(build).route("¿Qué requisitos tiene DEMO-202?")
    assert d.route == Route.STRUCTURED and "DEMO-202" in d.subjects


def test_clave_sin_guion_y_por_nombre(build):
    r = _router(build)
    assert "DEMO-201" in r.route("requisitos de demo 201").subjects
    assert "DEMO-201" in r.route("¿cuáles son las previas de bases de datos (ejemplo)?").subjects


def test_documental_por_defecto(build):
    assert _router(build).route("¿Cuántas veces puedo presentar examen?").route == Route.DOCUMENTAL


def test_fuera_de_alcance(build):
    assert _router(build).route("dame mi horoscopo").route == Route.OUT_OF_SCOPE
