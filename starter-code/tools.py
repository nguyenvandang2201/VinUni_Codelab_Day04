"""
Lab #4: Tool functions & JSON Schemas cho VinAssistant.

Học viên: Nguyễn Văn Đăng
MSSV: 2A202603371
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

# Trap 1: tính đường dẫn theo vị trí file này, không theo thư mục đang chạy lệnh.
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw-data")

VALID_CATEGORIES = ("xe_dien", "du_lich")
VALID_PRIORITIES = ("low", "medium", "high")
NO_PRICE_LIMIT = 999999999999
VIETNAM_TZ = timezone(timedelta(hours=7))


# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = NO_PRICE_LIMIT) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.

    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.

    Returns:
        Danh sách sản phẩm phù hợp, sắp xếp theo giá tăng dần (có thể rỗng).
        Nếu có lỗi, trả về [{"error": "..."}] để agent biết và báo lại cho khách.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    category = str(category or "").strip().lower()
    if category not in VALID_CATEGORIES:
        return [{"error": f"Danh mục '{category}' không hợp lệ. Chỉ hỗ trợ: {', '.join(VALID_CATEGORIES)}."}]

    if max_price is None:
        max_price = NO_PRICE_LIMIT
    try:
        max_price = int(max_price)
    except (TypeError, ValueError):
        return [{"error": f"max_price phải là số nguyên (VNĐ), nhận được: {max_price!r}."}]

    try:
        with open(catalog_file, "r", encoding="utf-8") as f:
            products = json.load(f)
    except json.JSONDecodeError as exc:
        return [{"error": f"File catalog sai định dạng JSON: {exc}"}]

    results = [
        p for p in products
        if str(p.get("category", "")).lower() == category
        and p.get("price_vnd") is not None
        and p["price_vnd"] <= max_price
    ]
    return sorted(results, key=lambda p: p["price_vnd"])


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.

    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
        Nếu dữ liệu đầu vào không hợp lệ, trả về {"error": "..."} và không ghi file.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    customer_name = str(customer_name or "").strip()
    issue_description = str(issue_description or "").strip()
    priority = str(priority or "medium").strip().lower()

    if not customer_name:
        return {"error": "Thiếu tên khách hàng (customer_name)."}
    if not issue_description:
        return {"error": "Thiếu mô tả vấn đề (issue_description)."}
    if priority not in VALID_PRIORITIES:
        return {"error": f"priority '{priority}' không hợp lệ. Chỉ nhận: {', '.join(VALID_PRIORITIES)}."}

    # Trap 2: luôn đọc ticket cũ trước, append, rồi mới ghi lại (không ghi đè).
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        try:
            with open(tickets_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            existing_tickets = json.loads(content) if content else []
        except json.JSONDecodeError as exc:
            # File hỏng thì dừng lại, không ghi đè để tránh mất ticket cũ.
            return {"error": f"File ticket sai định dạng JSON, không thể ghi thêm: {exc}"}
        if not isinstance(existing_tickets, list):
            return {"error": "File ticket không chứa danh sách ticket hợp lệ."}

    now = datetime.now(VIETNAM_TZ)
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{now.strftime('%Y%m%d')}-{seq:03d}"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": now.isoformat(),
        "category": "general",
    }
    existing_tickets.append(new_ticket)

    # Ghi ra file tạm rồi thay thế, để file gốc không bị hỏng nếu ghi giữa chừng thất bại.
    tmp_file = tickets_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=2, ensure_ascii=False)
    os.replace(tmp_file, tickets_file)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": new_ticket["created_at"],
        "message": f"Ticket {ticket_id} đã được tạo thành công.",
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS: JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, gói du lịch Vinpearl) theo danh mục "
            "và giá tối đa. Dùng khi khách hỏi về sản phẩm, giá hoặc tình trạng hàng. "
            "Trả về danh sách sản phẩm sắp xếp theo giá tăng dần (có thể rỗng)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' (xe điện VinFast) hoặc 'du_lich' (resort/khách sạn Vinpearl).",
                    "enum": ["xe_dien", "du_lich"],
                },
                "max_price": {
                    "type": "integer",
                    "description": (
                        "Giá tối đa tính bằng VNĐ, đã quy đổi về số nguyên "
                        "(VD: '600 triệu' -> 600000000). Bỏ trống nếu khách không nêu ngân sách."
                    ),
                    "minimum": 0,
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Tạo ticket hỗ trợ khi khách báo lỗi sản phẩm/dịch vụ, khiếu nại hoặc muốn ghi nhận phản hồi. "
            "Trả về ticket_id, mức ưu tiên và trạng thái ticket."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên khách hàng do chính khách cung cấp. Không được tự đoán.",
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả ngắn gọn vấn đề khách gặp phải.",
                },
                "priority": {
                    "type": "string",
                    "description": (
                        "Mức ưu tiên: 'high' nếu khách nói nghiêm trọng/gấp/khẩn cấp, "
                        "'low' nếu khách nói không gấp, còn lại 'medium'."
                    ),
                    "enum": ["low", "medium", "high"],
                },
            },
            "required": ["customer_name", "issue_description"],
        },
    },
]


# ---------------------------------------------------------------------------
# TOOL_MAP: Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
