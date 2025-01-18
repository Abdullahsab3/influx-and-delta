use arrow::buffer::{BooleanBuffer, Buffer};
use std::ops::Range;

/// An arrow-compatible mutable bitset implementation
///
/// Note: This currently operates on individual bytes at a time
/// it could be optimised to instead operate on usize blocks
#[derive(Debug, Default, Clone)]
pub struct BitSet {
    /// The underlying data
    ///
    /// Data is stored in the least significant bit of a byte first
    buffer: Vec<u8>,

    /// The length of this mask in bits
    len: usize,
}

impl BitSet {
    /// Creates a new BitSet
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct an empty [`BitSet`] with a pre-allocated capacity for `n`
    /// bits.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            buffer: Vec::with_capacity((n + 7) / 8),
            len: 0,
        }
    }

    /// Creates a new BitSet with `count` unset bits.
    pub fn with_size(count: usize) -> Self {
        let mut bitset = Self::default();
        bitset.append_unset(count);
        bitset
    }

    /// Reserve space for `count` further bits
    pub fn reserve(&mut self, count: usize) {
        let new_buf_len = (self.len + count + 7) / 8;
        self.buffer.reserve(new_buf_len);
    }

    /// Appends `count` unset bits
    pub fn append_unset(&mut self, count: usize) {
        self.len += count;
        let new_buf_len = (self.len + 7) / 8;
        self.buffer.resize(new_buf_len, 0);
    }

    /// Appends `count` set bits
    pub fn append_set(&mut self, count: usize) {
        let new_len = self.len + count;
        let new_buf_len = (new_len + 7) / 8;

        let skew = self.len % 8;
        if skew != 0 {
            *self.buffer.last_mut().unwrap() |= 0xFF << skew;
        }

        self.buffer.resize(new_buf_len, 0xFF);

        let rem = new_len % 8;
        if rem != 0 {
            *self.buffer.last_mut().unwrap() &= (1 << rem) - 1;
        }

        self.len = new_len;
    }

    /// Truncates the bitset to the provided length
    pub fn truncate(&mut self, len: usize) {
        let new_buf_len = (len + 7) / 8;
        self.buffer.truncate(new_buf_len);
        let overrun = len % 8;
        if overrun > 0 {
            *self.buffer.last_mut().unwrap() &= (1 << overrun) - 1;
        }
        self.len = len;
    }

    /// Split this bitmap at the specified bit boundary, such that after this
    /// call, `self` contains the range `[0, n)` and the returned value contains
    /// `[n, len)`.
    pub fn split_off(&mut self, n: usize) -> Self {
        let mut right = Self::with_capacity(self.len - n);
        right.extend_from_range(self, n..self.len);

        self.truncate(n);

        right
    }

    /// Extends this [`BitSet`] by the context of `other`
    pub fn extend_from(&mut self, other: &Self) {
        self.append_bits(other.len, &other.buffer)
    }

    /// Extends this [`BitSet`] by `range` elements in `other`
    pub fn extend_from_range(&mut self, other: &Self, range: Range<usize>) {
        let count = range.end - range.start;
        if count == 0 {
            return;
        }

        let start_byte = range.start / 8;
        let end_byte = (range.end + 7) / 8;
        let skew = range.start % 8;

        // `append_bits` requires the provided `to_set` to be byte aligned, therefore
        // if the range being copied is not byte aligned we must first append
        // the leading bits to reach a byte boundary
        if skew == 0 {
            // No skew can simply append bytes directly
            self.append_bits(count, &other.buffer[start_byte..end_byte])
        } else if start_byte + 1 == end_byte {
            // Append bits from single byte
            self.append_bits(count, &[other.buffer[start_byte] >> skew])
        } else {
            // Append trailing bits from first byte to reach byte boundary, then append
            // bits from the remaining byte-aligned mask
            let offset = 8 - skew;
            self.append_bits(offset, &[other.buffer[start_byte] >> skew]);
            self.append_bits(count - offset, &other.buffer[(start_byte + 1)..end_byte]);
        }
    }

    /// Appends `count` boolean values from the slice of packed bits
    pub fn append_bits(&mut self, count: usize, to_set: &[u8]) {
        assert_eq!((count + 7) / 8, to_set.len());

        let new_len = self.len + count;
        let new_buf_len = (new_len + 7) / 8;
        self.buffer.reserve(new_buf_len - self.buffer.len());

        let whole_bytes = count / 8;
        let overrun = count % 8;

        let skew = self.len % 8;
        if skew == 0 {
            self.buffer.extend_from_slice(&to_set[..whole_bytes]);
            if overrun > 0 {
                let masked = to_set[whole_bytes] & ((1 << overrun) - 1);
                self.buffer.push(masked)
            }

            self.len = new_len;
            debug_assert_eq!(self.buffer.len(), new_buf_len);
            return;
        }

        for to_set_byte in &to_set[..whole_bytes] {
            let low = *to_set_byte << skew;
            let high = *to_set_byte >> (8 - skew);

            *self.buffer.last_mut().unwrap() |= low;
            self.buffer.push(high);
        }

        if overrun > 0 {
            let masked = to_set[whole_bytes] & ((1 << overrun) - 1);
            let low = masked << skew;
            *self.buffer.last_mut().unwrap() |= low;

            if overrun > 8 - skew {
                let high = masked >> (8 - skew);
                self.buffer.push(high)
            }
        }

        self.len = new_len;
        debug_assert_eq!(self.buffer.len(), new_buf_len);
    }

    /// Sets a given bit
    pub fn set(&mut self, idx: usize) {
        assert!(idx <= self.len);

        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        self.buffer[byte_idx] |= 1 << bit_idx;
    }

    /// Returns if the given index is set
    pub fn get(&self, idx: usize) -> bool {
        assert!(idx <= self.len);

        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        (self.buffer[byte_idx] >> bit_idx) & 1 != 0
    }

    /// Converts this BitSet to a buffer compatible with arrows boolean
    /// encoding, consuming self.
    pub fn into_arrow(self) -> BooleanBuffer {
        let offset = 0;
        BooleanBuffer::new(Buffer::from_vec(self.buffer), offset, self.len)
    }

    /// Returns the number of values stored in the bitset
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns if this bitset is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of bytes used by this bitset
    pub fn byte_len(&self) -> usize {
        self.buffer.len()
    }

    /// Return the raw packed bytes used by this bitset
    pub fn bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Return `true` if all bits in the [`BitSet`] are currently set.
    pub fn is_all_set(&self) -> bool {
        // An empty bitmap has no set bits.
        if self.len == 0 {
            return false;
        }

        // Check all the bytes in the bitmap that have all their bits considered
        // part of the bit set.
        let full_blocks = (self.len / 8).saturating_sub(1);
        if !self.buffer.iter().take(full_blocks).all(|&v| v == u8::MAX) {
            return false;
        }

        // Check the last byte of the bitmap that may only be partially part of
        // the bit set, and therefore need masking to check only the relevant
        // bits.
        let mask = match self.len % 8 {
            1..=8 => !(0xFF << (self.len % 8)), // LSB mask
            0 => 0xFF,
            _ => unreachable!(),
        };
        *self.buffer.last().unwrap() == mask
    }

    /// Return `true` if all bits in the [`BitSet`] are currently unset.
    pub fn is_all_unset(&self) -> bool {
        self.buffer.iter().all(|&v| v == 0)
    }

    /// Returns the number of set bits in this bitmap.
    pub fn count_ones(&self) -> usize {
        // Invariant: the bits outside of [0, self.len) are always 0
        self.buffer.iter().map(|v| v.count_ones() as usize).sum()
    }

    /// Returns the number of unset bits in this bitmap.
    pub fn count_zeros(&self) -> usize {
        self.len() - self.count_ones()
    }

    /// Returns true if any bit is set (short circuiting).
    pub fn is_any_set(&self) -> bool {
        self.buffer.iter().any(|&v| v != 0)
    }

    /// Returns a value [`Iterator`] that yields boolean values encoded in the
    /// bitmap.
    pub fn iter(&self) -> Iter<'_> {
        Iter::new(self)
    }

    /// Returns the bitwise AND between the two [`BitSet`] instances.
    ///
    /// # Panics
    ///
    /// Panics if the two sets have differing lengths.
    pub fn and(&self, other: &Self) -> Self {
        assert_eq!(self.len, other.len);

        Self {
            buffer: self
                .buffer
                .iter()
                .zip(other.buffer.iter())
                .map(|(a, b)| a & b)
                .collect(),
            len: self.len,
        }
    }
}

/// A value iterator yielding the boolean values encoded in the bitmap.
#[derive(Debug)]
pub struct Iter<'a> {
    /// A reference to the bitmap buffer.
    buffer: &'a [u8],
    /// The index of the next yielded bit in `buffer`.
    idx: usize,
    /// The number of bits stored in buffer.
    len: usize,
}

impl<'a> Iter<'a> {
    fn new(b: &'a BitSet) -> Self {
        Self {
            buffer: &b.buffer,
            idx: 0,
            len: b.len(),
        }
    }
}

impl Iterator for Iter<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            return None;
        }

        let byte_idx = self.idx / 8;
        let shift = self.idx % 8;

        self.idx += 1;

        let byte = self.buffer[byte_idx];
        let byte = byte >> shift;

        Some(byte & 1 == 1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let v = self.len - self.idx;
        (v, Some(v))
    }
}

impl ExactSizeIterator for Iter<'_> {}

/// Returns an iterator over set bit positions in increasing order
pub fn iter_set_positions(bytes: &[u8]) -> impl Iterator<Item = usize> + '_ {
    iter_set_positions_with_offset(bytes, 0)
}

/// Returns an iterator over set bit positions in increasing order starting
/// at the provided bit offset
pub fn iter_set_positions_with_offset(
    bytes: &[u8],
    offset: usize,
) -> impl Iterator<Item = usize> + '_ {
    let mut byte_idx = offset / 8;
    let mut in_progress = bytes.get(byte_idx).cloned().unwrap_or(0);

    let skew = offset % 8;
    in_progress &= 0xFF << skew;

    std::iter::from_fn(move || loop {
        if in_progress != 0 {
            let bit_pos = in_progress.trailing_zeros();
            in_progress ^= 1 << bit_pos;
            return Some((byte_idx * 8) + (bit_pos as usize));
        }
        byte_idx += 1;
        in_progress = *bytes.get(byte_idx)?;
    })
}