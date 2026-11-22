

def test_ruta_del_manifiesto_con_acentos_descompuestos(tmp_path):
    import unicodedata
    from chatbot.ingestion.build_index import resolve_path
    real = tmp_path / unicodedata.normalize("NFD", "Programa EE_Programación.pdf")
    real.write_bytes(b"x")
    pedido = tmp_path / unicodedata.normalize("NFC", "Programa EE_Programación.pdf")
    assert resolve_path(pedido).read_bytes() == b"x"
    assert resolve_path(tmp_path / "no-existe.pdf") == tmp_path / "no-existe.pdf"
