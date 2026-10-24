"""Worker long-polling cho Zalo Bot (GIAI ĐOẠN 1).

Chạy: `python -m app.integrations.zalo_poll`

Gọi `getUpdates` liên tục và điều phối mỗi update qua `_process_zalo_update`. KHÔNG cần URL HTTPS công khai — hợp cho máy local sau NAT. Webhook &
polling loại trừ lẫn nhau theo API Zalo nên worker gọi `deleteWebhook` trước.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import ZALO_BOT_TOKEN, ZALO_ENABLE_POLLING, ZALO_POLL_TIMEOUT

logger = logging.getLogger(__name__)


def run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not ZALO_BOT_TOKEN:
        logger.error("Thiếu ZALO_BOT_TOKEN trong .env — không thể polling.")
        return 1
    if not ZALO_ENABLE_POLLING:
        logger.error("ZALO_ENABLE_POLLING chưa bật (đặt ZALO_ENABLE_POLLING=1 trong .env).")
        return 1

    # Import ở đây (không ở top) để thông báo cấu hình thiếu nhanh, trước khi khởi tạo agent
    # (việc import app.api sẽ tạo agent một lần — cần cho trả lời).
    from app.api import _process_zalo_update, zalo_client

    # Webhook & polling loại trừ lẫn nhau: gỡ webhook (nếu có) trước khi poll.
    zalo_client.delete_webhook()

    me = zalo_client.get_me()
    if me is None:
        logger.error("getMe thất bại — kiểm tra token/kết nối. Vẫn tiếp tục thử polling.")
    else:
        logger.info("Zalo bot: %s", me)

    logger.info("Bắt đầu long-polling (timeout=%ss). Ctrl+C để dừng.", ZALO_POLL_TIMEOUT)
    while True:
        try:
            updates = zalo_client.get_updates(ZALO_POLL_TIMEOUT)
            for update in updates:
                _process_zalo_update(update)
        except KeyboardInterrupt:
            logger.info("Dừng polling.")
            return 0
        except Exception as exc:  # noqa: BLE001 - không để vòng lặp chết vì một lỗi tạm thời
            logger.warning("Vòng polling lỗi, thử lại sau 3s: %s", exc)
            time.sleep(3)

async def poll_zalo_updates() -> None:
    """Chạy Long Polling liên tục mà không block event loop của FastAPI."""
    from app.api import _process_zalo_update, zalo_client

    while True:
        try:
            # get_updates() là hàm đồng bộ; đưa sang thread riêng.
            updates = await asyncio.to_thread(
                zalo_client.get_updates,
                ZALO_POLL_TIMEOUT,
            )

            for update in updates:
                try:
                    # Không block event loop khi xử lý update/agent.
                    await asyncio.to_thread(
                        _process_zalo_update,
                        update,
                    )
                except Exception:
                    logger.exception(
                        "Không xử lý được một Zalo update: %s",
                        update,
                    )

        except asyncio.CancelledError:
            logger.info("Nhận yêu cầu dừng Zalo polling.")
            raise

        except Exception:
            logger.exception("Vòng polling lỗi; thử lại sau 3 giây.")
            await asyncio.sleep(3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo bot khi ứng dụng chạy và dừng polling khi shutdown."""
    if not ZALO_BOT_TOKEN:
        raise RuntimeError("Thiếu ZALO_BOT_TOKEN trong môi trường.")

    if not ZALO_ENABLE_POLLING:
        raise RuntimeError(
            "ZALO_ENABLE_POLLING chưa bật. "
            "Hãy đặt ZALO_ENABLE_POLLING=1."
        )

    # Import tại startup để agent/client chỉ được khởi tạo khi cần.
    from app.api import zalo_client

    try:
        # Long Polling và Webhook không nên hoạt động đồng thời.
        await asyncio.to_thread(zalo_client.delete_webhook)

        me = await asyncio.to_thread(zalo_client.get_me)
        if me is None:
            logger.error(
                "getMe thất bại; polling vẫn sẽ thử kết nối."
            )
        else:
            logger.info("Zalo bot: %s", me)

        polling_task = asyncio.create_task(
            poll_zalo_updates(),
            name="zalo-long-polling",
        )
        app.state.polling_task = polling_task

        logger.info(
            "Bắt đầu Zalo Long Polling (timeout=%ss).",
            ZALO_POLL_TIMEOUT,
        )

        yield

    finally:
        task = getattr(app.state, "polling_task", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info("Zalo polling đã dừng.")


app = FastAPI(
    title="Zalo AI Agent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check cho Docker và hệ thống giám sát."""
    task = getattr(app.state, "polling_task", None)
    polling_alive = task is not None and not task.done()

    if polling_alive:
        return {
            "status": "ok",
            "polling": "running",
        }

    return JSONResponse(
        status_code=503,
        content={
            "status": "unavailable",
            "polling": "stopped",
        },
    )



if __name__ == "__main__":
    # raise SystemExit(run())

    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
    )
