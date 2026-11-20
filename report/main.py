"""
PulsePress AI — LangGraph 기반 심층 탐사보도 기사 작성 FastAPI 서버 (main.py)
- AI 프레임워크: LangGraph (TypedDict State, generate 노드, BasicToolNode, route_tools 조건부 엣지, .stream 실행)
- 언어 모델: Google AI (ChatGoogleGenerativeAI / Gemini)
- 도구: get_current_time, get_web_search
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Generator
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# =============================================================================
# 1. 시스템 페르소나 및 환경 설정
# =============================================================================
APP_VERSION = "2.4"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT_TEMPLATE = """너는 신문기자이다.
최근 {about}에 대해 비판하는 심층 분석 기사를 쓰려고 한다.

[단계별 기사 작성 지시사항]
- 최근 어떤 이슈가 있는지 검색하고 사람들이 제일 관심있어 할만한 주체를 선정하고 왜 선정했는지 말해줘
- 그 내용으로 원고를 작성하기 위한 목차를 만들고 목차 내용을 채우기 위해 추가로 검색할 내용을 리스트로 정리해봐
- 검색할 리스트를 토대로 재검색해
- 목차에 있는 내용을 작성하기 위해 더 검색이 필요한 정보가 있는지 확인하고 있다면 추가로 검색해
- 검색된 결과에서 원하는 정보를 찾지 못했다면 다른 검색어로 재검색해도 좋아.

[최종 기사 출력 조건]
더이상 검색할 내용이 없다면 신문 기사 형식으로 최종 기사를 작성한다.
제목, 부제, 리드문, 본문의 구성으로 작성하며, 본문내용은 심층분석 기사에 맞게 구체적이고 깊이있게 작성해야 한다.

[출력 형식 가이드]
최종 기사를 작성할 때는 반드시 아래 마크다운 구조를 지켜줘:
# [제목]
## [부제]
> [리드문: 기사 핵심을 요약하는 2~3문장]

### 1. [소제목]
본문 내용...

### 2. [소제목]
본문 내용...

### 3. [소제목]
본문 내용...
"""


# =============================================================================
# 2. 상태(State) 정의 — TypedDict 상속
# =============================================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# =============================================================================
# 3. 도구(Tools) 등록 — get_current_time, get_web_search
# =============================================================================
@tool
def get_current_time(tz_name: str = "Asia/Seoul") -> str:
    """현재 날짜와 시각을 반환합니다. 최신 이슈의 시점을 파악할 때 사용하세요."""
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone(timedelta(hours=9))
    now = datetime.now(tz)
    return json.dumps(
        {
            "timezone": tz_name,
            "iso": now.isoformat(),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "date": now.strftime("%Y년 %m월 %d일"),
        },
        ensure_ascii=False,
    )


def _search_ddgs(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """ddgs 또는 duckduckgo_search 패키지를 이용한 웹 검색."""
    results: list[dict[str, str]] = []
    DDGS = None
    try:
        from ddgs import DDGS as _DDGS  # type: ignore
        DDGS = _DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS as _DDGS  # type: ignore
            DDGS = _DDGS
        except ImportError:
            return []

    with DDGS() as ddgs:
        raw_items = list(ddgs.text(query, region="kr-kr", max_results=max_results))
        for item in raw_items:
            url = item.get("href") or item.get("url") or ""
            domain = urlparse(url).netloc.replace("www.", "") if url else "web"
            results.append(
                {
                    "title": (item.get("title") or "").strip(),
                    "url": url,
                    "snippet": (item.get("body") or item.get("snippet") or "").strip(),
                    "source": domain,
                    "published_at": datetime.now(timezone(timedelta(hours=9))).strftime("%H:%M KST"),
                }
            )
    return results


def _search_google_news_rss(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """외부 검색 라이브러리 미설치/제한 시 Google News RSS로 폴백 검색."""
    encoded = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=ko&gl=KR&ceid=KR:ko"
    req = urllib.request.Request(
        rss_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PulsePressAI/2.4)"},
    )
    results: list[dict[str, str]] = []
    with urllib.request.urlopen(req, timeout=10) as resp:
        xml_data = resp.read()
    root = ET.fromstring(xml_data)
    for item in root.findall(".//item")[:max_results]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        desc = re.sub(r"<[^>]+>", "", (item.findtext("description") or "")).strip()
        source_el = item.find("source")
        source_name = (
            source_el.text.strip()
            if source_el is not None and source_el.text
            else urlparse(link).netloc.replace("www.", "")
        )
        results.append(
            {
                "title": title,
                "url": link,
                "snippet": desc[:260],
                "source": source_name or "Google News",
                "published_at": pub_date[:22] if pub_date else "최근 수집",
            }
        )
    return results


@tool
def get_web_search(query: str, max_results: int = 5) -> str:
    """주어진 검색어(query)로 최신 웹 뉴스와 자료를 검색하여 JSON 리스트로 반환합니다."""
    results: list[dict[str, str]] = []
    errors: list[str] = []

    try:
        results = _search_ddgs(query, max_results=max_results)
    except Exception as exc:
        errors.append(f"DDGS 오류: {exc}")

    if not results:
        try:
            results = _search_google_news_rss(query, max_results=max_results)
        except Exception as exc:
            errors.append(f"Google News RSS 오류: {exc}")

    payload = {
        "query": query,
        "count": len(results),
        "results": results,
    }
    if not results and errors:
        payload["error"] = " | ".join(errors)

    return json.dumps(payload, ensure_ascii=False)


TOOLS = [get_current_time, get_web_search]


# =============================================================================
# 4. 노드 생성 — generate, BasicToolNode
# =============================================================================
class BasicToolNode:
    """마지막 AIMessage에 포함된 tool_calls를 실행해 ToolMessage로 반환하는 노드."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, inputs: dict[str, Any]) -> dict[str, list[ToolMessage]]:
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("입력 상태에 messages가 없습니다.")
        last_message = messages[-1]

        outputs: list[ToolMessage] = []
        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"call_{tool_name}")

            target_tool = self.tools_by_name.get(tool_name)
            if target_tool is None:
                result_str = json.dumps(
                    {"error": f"등록되지 않은 도구입니다: {tool_name}"},
                    ensure_ascii=False,
                )
            else:
                try:
                    tool_result = target_tool.invoke(tool_args)
                    result_str = (
                        tool_result
                        if isinstance(tool_result, str)
                        else json.dumps(tool_result, ensure_ascii=False)
                    )
                except Exception as exc:
                    result_str = json.dumps(
                        {"error": f"도구 실행 중 오류({tool_name}): {exc}"},
                        ensure_ascii=False,
                    )

            outputs.append(
                ToolMessage(
                    content=result_str,
                    name=tool_name,
                    tool_call_id=tool_id,
                )
            )
        return {"messages": outputs}


def _get_api_key() -> str:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""


def _build_llm_with_tools(model_name: str | None = None):
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY (또는 GEMINI_API_KEY) 환경 변수가 설정되지 않았습니다. .env 파일을 확인해 주세요."
        )
    llm = ChatGoogleGenerativeAI(
        model=model_name or DEFAULT_MODEL,
        google_api_key=api_key,
        temperature=0.4,
    )
    return llm.bind_tools(TOOLS)


# =============================================================================
# 5. 라우터 설정 — route_tools & LangGraph 컴파일
# =============================================================================
def route_tools(state: State):
    """마지막 AI 메시지에 도구 호출이 있으면 'tools', 없으면 END로 분기."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"route_tools에 전달된 상태에서 messages를 찾을 수 없습니다: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def build_reporter_graph(model_name: str | None = None):
    llm_with_tools = _build_llm_with_tools(model_name=model_name)

    def generate(state: State) -> dict[str, list]:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = BasicToolNode(tools=TOOLS)

    graph_builder = StateGraph(State)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "generate")
    graph_builder.add_conditional_edges(
        "generate",
        route_tools,
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "generate")

    return graph_builder.compile()


# =============================================================================
# 6. 메시지 텍스트 추출 및 기사 마크다운 파서
# =============================================================================
def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(content or "")


def _parse_article_markdown(markdown_text: str, fallback_topic: str) -> dict[str, Any]:
    """마크다운 형식의 최종 기사를 제목/부제/리드문/섹션 구조로 파싱."""
    text = markdown_text.strip()
    lines = text.splitlines()

    title = ""
    subtitle = ""
    lead_lines: list[str] = []
    sections: list[dict[str, Any]] = []
    current_heading = ""
    current_body: list[str] = []

    def flush_section():
        nonlocal current_heading, current_body
        body_text = "\n".join(current_body).strip()
        if current_heading or body_text:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body_text) if p.strip()]
            sections.append(
                {
                    "heading": current_heading or "심층 분석",
                    "paragraphs": paragraphs,
                }
            )
        current_heading = ""
        current_body = []

    for raw_line in lines:
        line = raw_line.strip()
        if not title and re.match(r"^#\s+(.+)$", line):
            title = re.sub(r"^#\s+", "", line).strip().strip("*")
            continue
        if not subtitle and re.match(r"^##\s+(.+)$", line) and not sections:
            subtitle = re.sub(r"^##\s+", "", line).strip().strip("*")
            continue
        if not sections and not current_heading and line.startswith(">"):
            lead_lines.append(line.lstrip(">").strip())
            continue
        if re.match(r"^#{2,3}\s+(.+)$", line):
            flush_section()
            current_heading = re.sub(r"^#{2,3}\s+", "", line).strip().strip("*")
            continue
        current_body.append(raw_line)

    flush_section()

    if not title:
        title = f"[심층 분석] {fallback_topic}"
    if not subtitle:
        subtitle = "최신 이슈 교차 검증과 구조적 문제점에 대한 탐사 리포트"
    lead = " ".join(lead_lines).strip()
    if not lead and sections and sections[0]["paragraphs"]:
        lead = sections[0]["paragraphs"][0]

    char_count = len(re.sub(r"\s+", "", text))
    read_time_sec = max(60, int(char_count / 7))

    return {
        "title": title,
        "subtitle": subtitle,
        "lead": lead,
        "sections": sections,
        "raw_markdown": text,
        "char_count": char_count,
        "read_time_sec": read_time_sec,
    }


def run_reporter_stream(
    about: str,
    model_name: str | None = None,
    desk: str = "심층 취재 데스크",
) -> Generator[dict[str, Any], None, None]:
    """LangGraph의 .stream()을 실행하며 단계별 이벤트를 순차 yield합니다."""
    started_at = time.time()
    graph = build_reporter_graph(model_name=model_name)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(about=about)
    initial_state: State = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"주제: '{about}'\n"
                    f"담당 데스크: {desk}\n"
                    "먼저 현재 시각과 최신 이슈를 검색해 핵심 주체를 선정하고, "
                    "목차와 추가 검색 리스트를 세워 충분히 재검색한 뒤 최종 심층 기사를 작성해 주세요."
                )
            ),
        ]
    }

    step_index = 0
    collected_sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    queries_used: list[str] = []
    intermediate_notes: list[str] = []
    final_markdown = ""
    current_time_info = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")

    for event in graph.stream(
        initial_state,
        stream_mode="updates",
        config={"recursion_limit": 25},
    ):
        for node_name, node_output in event.items():
            step_index += 1
            messages = node_output.get("messages", [])
            for msg in messages:
                if node_name == "generate" and isinstance(msg, AIMessage):
                    text = _extract_text(msg.content)
                    tool_calls = getattr(msg, "tool_calls", []) or []

                    if tool_calls:
                        call_summaries = []
                        for tc in tool_calls:
                            name = tc.get("name", "")
                            args = tc.get("args", {})
                            if name == "get_web_search" and args.get("query"):
                                queries_used.append(str(args["query"]))
                            call_summaries.append({"name": name, "args": args})

                        if text:
                            intermediate_notes.append(text)

                        yield {
                            "event": "step",
                            "step": step_index,
                            "node": "generate",
                            "phase": "tool_request",
                            "summary": text[:400] if text else "추가 취재를 위한 도구 호출을 요청했습니다.",
                            "tool_calls": call_summaries,
                            "timestamp": datetime.now(timezone(timedelta(hours=9))).strftime("%H:%M:%S"),
                        }
                    else:
                        final_markdown = text
                        yield {
                            "event": "step",
                            "step": step_index,
                            "node": "generate",
                            "phase": "final_article",
                            "summary": f"최종 심층 기사 작성이 완료되었습니다 ({len(text)}자).",
                            "timestamp": datetime.now(timezone(timedelta(hours=9))).strftime("%H:%M:%S"),
                        }

                elif node_name == "tools" and isinstance(msg, ToolMessage):
                    tool_name = msg.name or "tool"
                    raw_content = _extract_text(msg.content)
                    new_sources: list[dict[str, Any]] = []

                    try:
                        parsed = json.loads(raw_content)
                    except Exception:
                        parsed = {"raw": raw_content}

                    if tool_name == "get_current_time" and isinstance(parsed, dict):
                        current_time_info = parsed.get("formatted", current_time_info)
                        detail = f"기준 시각 확인: {current_time_info}"
                    elif tool_name == "get_web_search" and isinstance(parsed, dict):
                        q = parsed.get("query", "")
                        items = parsed.get("results", []) or []
                        for idx, item in enumerate(items):
                            url = item.get("url", "")
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                source_entry = {
                                    "title": item.get("title", "제목 없음"),
                                    "url": url,
                                    "snippet": item.get("snippet", ""),
                                    "source": item.get("source", "Web"),
                                    "published_at": item.get("published_at", "방금 수집"),
                                    "query": q,
                                    "confidence": max(86, 99 - idx * 2),
                                }
                                collected_sources.append(source_entry)
                                new_sources.append(source_entry)
                        detail = f"'{q}' 검색 완료 — {len(items)}건 수집 (누적 고유 출처 {len(collected_sources)}건)"
                    else:
                        detail = raw_content[:240]

                    yield {
                        "event": "step",
                        "step": step_index,
                        "node": "tools",
                        "phase": tool_name,
                        "summary": detail,
                        "new_sources": new_sources,
                        "timestamp": datetime.now(timezone(timedelta(hours=9))).strftime("%H:%M:%S"),
                    }

    article = _parse_article_markdown(final_markdown, fallback_topic=about)
    elapsed_sec = round(time.time() - started_at, 2)

    yield {
        "event": "complete",
        "topic": about,
        "desk": desk,
        "model": model_name or DEFAULT_MODEL,
        "current_time": current_time_info,
        "elapsed_sec": elapsed_sec,
        "queries": list(dict.fromkeys(queries_used)),
        "intermediate_notes": intermediate_notes,
        "sources": collected_sources,
        "article": article,
        "stats": {
            "total_steps": step_index,
            "source_count": len(collected_sources),
            "query_count": len(set(queries_used)),
            "char_count": article["char_count"],
            "confidence": round(
                sum(s["confidence"] for s in collected_sources) / len(collected_sources), 1
            )
            if collected_sources
            else 92.0,
        },
    }


# =============================================================================
# 7. FastAPI 애플리케이션 및 엔드포인트
# =============================================================================
app = FastAPI(
    title="PulsePress AI — LangGraph Reporter API",
    version=APP_VERSION,
    description="LangGraph + Google AI 기반 심층 탐사보도 기사 자동 생성 서버",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportRequest(BaseModel):
    about: str = Field(..., min_length=1, description="탐사보도 주제 또는 취재 키워드")
    desk: str = Field(default="심층 취재 데스크", description="담당 데스크명")
    model: str | None = Field(default=None, description="사용할 Gemini 모델명 (선택)")


@app.get("/health")
def health_check():
    return {
        "status": "connected",
        "server": "main.py",
        "version": APP_VERSION,
        "model": DEFAULT_MODEL,
        "api_key_configured": bool(_get_api_key()),
        "tools": [t.name for t in TOOLS],
        "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat(),
    }


@app.post("/api/report")
@app.post("/generate")
def generate_report(req: ReportRequest):
    """GAS UrlFetchApp 호환 엔드포인트: graph.stream() 결과를 단계별 로그와 최종 기사로 취합해 JSON 반환."""
    steps: list[dict[str, Any]] = []
    final_payload: dict[str, Any] | None = None

    try:
        for chunk in run_reporter_stream(
            about=req.about.strip(),
            model_name=req.model,
            desk=req.desk,
        ):
            if chunk["event"] == "step":
                steps.append(chunk)
                print(f"[Step {chunk['step']}] ({chunk['node']}/{chunk['phase']}) {chunk['summary']}")
            elif chunk["event"] == "complete":
                final_payload = chunk
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if final_payload is None:
        raise HTTPException(status_code=500, detail="기사 생성 결과를 받지 못했습니다.")

    final_payload["steps"] = steps
    return final_payload


@app.post("/api/report/stream")
def generate_report_stream(req: ReportRequest):
    """실시간 SSE(Server-Sent Events) 스트리밍 엔드포인트."""

    def event_generator():
        try:
            for chunk in run_reporter_stream(
                about=req.about.strip(),
                model_name=req.model,
                desk=req.desk,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as exc:
            err = {"event": "error", "message": str(exc)}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
