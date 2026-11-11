// 4-lane 8-bit SIMD unit (held-out design): per-lane add, sub, low-byte multiply,
// saturating add, max, and a horizontal sum of the four products.
module simd4 (
    input  wire        clk,
    input  wire        rst,
    input  wire [2:0]  op,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] y,
    output reg  [17:0] hsum
);
    reg [2:0]  rop;
    reg [31:0] ra, rb;
    reg [31:0] r;
    reg [15:0] p [0:3];
    integer k;
    always @(*) begin
        for (k = 0; k < 4; k = k + 1) begin
            p[k] = ra[8*k +: 8] * rb[8*k +: 8];
            case (rop)
                3'd0: r[8*k +: 8] = ra[8*k +: 8] + rb[8*k +: 8];
                3'd1: r[8*k +: 8] = ra[8*k +: 8] - rb[8*k +: 8];
                3'd2: r[8*k +: 8] = p[k][7:0];
                3'd3: r[8*k +: 8] = ({1'b0, ra[8*k +: 8]} + rb[8*k +: 8] > 9'd255) ? 8'hff
                                    : ra[8*k +: 8] + rb[8*k +: 8];
                default: r[8*k +: 8] = (ra[8*k +: 8] > rb[8*k +: 8]) ? ra[8*k +: 8] : rb[8*k +: 8];
            endcase
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            rop <= 0; ra <= 0; rb <= 0; y <= 0; hsum <= 0;
        end else begin
            rop <= op; ra <= a; rb <= b; y <= r;
            hsum <= p[0] + p[1] + p[2] + p[3];
        end
    end
endmodule
