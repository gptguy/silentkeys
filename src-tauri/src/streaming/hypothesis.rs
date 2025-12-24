pub const FRAME_DURATION_MS: i64 = 80;

#[derive(Debug, Clone, PartialEq)]
pub struct TokenWithTime {
    pub token_id: i32,
    pub text: String,
    pub start_frame: usize,
    pub end_frame: usize,
}

impl TokenWithTime {
    pub fn start_ms(&self) -> i64 {
        (self.start_frame as i64) * FRAME_DURATION_MS
    }

    pub fn end_ms(&self) -> i64 {
        (self.end_frame as i64) * FRAME_DURATION_MS
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WordWithTime {
    pub text: String,
    pub t0_ms: i64,
    pub t1_ms: i64,
    pub end_frame: usize,
    pub last_token_id: i32,
}

impl WordWithTime {
    pub fn center_ms(&self) -> i64 {
        (self.t0_ms + self.t1_ms) / 2
    }

    pub fn matches_in_time_bucket(&self, other: &WordWithTime, bucket_ms: i64) -> bool {
        if self.text != other.text {
            return false;
        }
        let self_bucket = self.center_ms() / bucket_ms;
        let other_bucket = other.center_ms() / bucket_ms;
        self_bucket == other_bucket
    }
}
