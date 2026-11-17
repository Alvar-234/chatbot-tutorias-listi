"""Verificación TLS con el almacén de certificados del sistema operativo.

Necesario cuando un antivirus o proxy (p. ej. Avast Web Shield) inspecciona HTTPS con una raíz
propia que el sistema sí reconoce pero `certifi` no: sin esto fallan Groq, Hugging Face, etc.
La verificación NUNCA se desactiva; si `truststore` no está instalado no se cambia nada.
"""
from __future__ import annotations


def use_system_certificates() -> bool:
    try:
        import truststore
    except ImportError:
        return False
    truststore.inject_into_ssl()
    return True
