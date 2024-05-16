import { createBrowserClient } from "@supabase/ssr";

export const createClient = () =>
  createBrowserClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_KEY!,
  );
