// 12x12 signed multiply-accumulate, 32-bit accumulator (training variant of mac16).
module mac12 (
    input  wire               clk,
    input  wire               rst,
    input  wire               en,
    input  wire signed [11:0] a,
    input  wire signed [11:0] b,
    output reg  signed [31:0] acc
);
    reg signed [11:0] ra, rb;
    reg               ren;
    always @(posedge clk) begin
        if (rst) begin
            ra <= 0; rb <= 0; ren <= 0; acc <= 0;
        end else begin
            ra <= a; rb <= b; ren <= en;
            if (ren) acc <= acc + ra * rb;
        end
    end
endmodule
