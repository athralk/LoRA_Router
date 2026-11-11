// 16x16 signed multiply-accumulate, 40-bit accumulator, registered in and out.
module mac16 (
    input  wire               clk,
    input  wire               rst,
    input  wire               en,
    input  wire signed [15:0] a,
    input  wire signed [15:0] b,
    output reg  signed [39:0] acc
);
    reg signed [15:0] ra, rb;
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
