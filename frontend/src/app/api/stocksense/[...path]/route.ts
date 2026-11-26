import { NextRequest, NextResponse } from "next/server";

type RouteContext = {
  params: Promise<{
    path: string[];
  }>;
};

export async function GET(
  request: NextRequest,
  context: RouteContext
) {
  const baseUrl = process.env.STOCKSENSE_API_URL;

  if (!baseUrl) {
    return NextResponse.json(
      { detail: "Backend API URL is not configured." },
      { status: 500 }
    );
  }

  try {
    const { path } = await context.params;
    const endpoint = path.join("/");
    const query = request.nextUrl.search;

    const upstreamUrl = `${baseUrl.replace(/\/+$/, "")}/${endpoint}${query}`;

    const upstreamResponse = await fetch(upstreamUrl, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      cache: "no-store",
      signal: AbortSignal.timeout(45000),
    });

    const responseBody = await upstreamResponse.text();

    return new NextResponse(responseBody, {
      status: upstreamResponse.status,
      headers: {
        "Content-Type":
          upstreamResponse.headers.get("content-type") ??
          "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    console.error("StockSense API proxy error:", error);

    return NextResponse.json(
      {
        detail:
          "Unable to reach the StockSense backend. It may be starting up or temporarily unavailable.",
      },
      { status: 502 }
    );
  }
}