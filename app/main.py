"""Worker long-polling cho Zalo Bot (GIAI ĐOẠN 1).

Chạy: `python -m app.integrations.zalo_poll`

Gọi `getUpdates` liên tục và điều phối mỗi update qua `_process_zalo_update` (tái dùng pipeline
RAG trong app/api.py). KHÔNG cần URL HTTPS công khai — hợp cho máy local sau NAT. Webhook &
polling loại trừ lẫn nhau theo API Zalo nên worker gọi `deleteWebhook` trước.
"""

import logging
import time

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


if __name__ == "__main__":
    raise SystemExit(run())
