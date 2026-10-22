import asyncio
import time
import httpx


URL = "http://127.0.0.1:8000/tasks"

CONSULTAS = [
    "¿Qué es Minecraft?",
    "¿Qué juego lanzo Capcom en el 2005?",
    "¿Qué es una consola?",
    "¿Qué es un RPG?",
    "¿Qué es un motor gráfico?",
]


async def enviar_tarea(client, consulta):
    inicio = time.perf_counter()

    response = await client.post(
        URL,
        json={"query": consulta},
    )

    datos = response.json()
    job_id = datos["job_id"]

    limite = time.monotonic() + 300
    while time.monotonic() < limite:
        response = await client.get(f"http://127.0.0.1:8000/tasks/{job_id}")
        response.raise_for_status()
        estado = response.json()
        # Los estados de espera también son finales para esta medición:
        # el script no responde aclaraciones ni aprobaciones interactivas.
        if estado.get("status") in [
            "DONE", "FAILED", "RETRYABLE", "WAITING_CLARIFICATION", "WAITING_APPROVAL", "REJECTED"
        ]:
            break
        await asyncio.sleep(0.5)
    else:
        estado = {"status": "TIMEOUT", "error": "La tarea no terminó en 300 segundos"}

    fin = time.perf_counter()

    return {
        "job_id": job_id,
        "status": estado["status"],
        "error": estado.get("error"),
        "clarification_question": estado.get("clarification_question"),
        "tiempo": fin - inicio,
    }


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        inicio_total = time.perf_counter()

        resultados = await asyncio.gather(
            *(enviar_tarea(client, consulta) for consulta in CONSULTAS)
        )

        fin_total = time.perf_counter()

    print("\nRESULTADOS:")
    for resultado in resultados:
        print(resultado)

    tiempos = sorted(resultado["tiempo"] for resultado in resultados)

    posicion = int(0.95 * len(tiempos)) - 1
    posicion = max(0, min(posicion, len(tiempos) - 1))

    p95 = tiempos[posicion]

    print(f"\nTiempo total: {fin_total - inicio_total:.2f}s")
    print(f"P95: {p95:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())