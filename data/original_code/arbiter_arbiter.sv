module arbiter(clk, sel, active);
  input clk, active;
  output reg [1:0] sel;
  parameter A = 0;
  parameter B = 1;
  parameter C = 2;
  parameter X = 3;
  reg [1:0] state;
  initial state = A;
  assign sel = active ? state : X;
  always @(posedge clk) begin
    if (active) begin
      case(state) 
        A: state = B;
        B: state = C;
        C: state = A;
      endcase
    end
  end
endmodule
