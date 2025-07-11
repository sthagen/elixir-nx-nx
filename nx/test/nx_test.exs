defmodule NxTest do
  use ExUnit.Case, async: true

  import Nx.Helpers
  import Nx, only: :sigils

  defp commute(a, b, fun) do
    fun.(a, b)
    fun.(b, a)
  end

  @doctypes [
    :aggregation,
    :backend,
    :conversion,
    :creation,
    :cumulative,
    :element,
    :indexed,
    :ndim,
    :shape,
    :type,
    :window,
    :vectorization
  ]

  test "defines doc metadata" do
    {:docs_v1, _, :elixir, "text/markdown", _docs, _metadata, entries} = Code.fetch_docs(Nx)

    for {{:function, name, arity}, _ann, _signature, docs, metadata} <- entries,
        is_map(docs) and map_size(docs) > 0,
        metadata[:type] not in @doctypes do
      flunk("invalid @doc type: #{inspect(metadata[:type])} for #{name}/#{arity}")
    end
  end

  describe "binary broadcast" do
    test "{2, 1} + {1, 2}" do
      commute(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<11::32-native, 21::32-native, 12::32-native, 22::32-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end

    test "{2} + {2, 2}" do
      commute(Nx.tensor([1, 2]), Nx.tensor([[1, 2], [3, 4]]), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<2::32-native, 4::32-native, 4::32-native, 6::32-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end

    test "{2, 1, 2} + {1, 2, 1}" do
      a = Nx.tensor([[[10], [20]]])
      assert Nx.shape(a) == {1, 2, 1}
      b = Nx.tensor([[[1, 2]], [[3, 4]]])
      assert Nx.shape(b) == {2, 1, 2}

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)
        assert Nx.shape(t) == {2, 2, 2}

        assert Nx.to_binary(t) ==
                 <<11::32-native, 12::32-native, 21::32-native, 22::32-native, 13::32-native,
                   14::32-native, 23::32-native, 24::32-native>>
      end)
    end

    test "{4, 1, 3} + {1, 3, 1}" do
      a = Nx.tensor([[[100], [200], [300]]])
      assert Nx.shape(a) == {1, 3, 1}
      b = Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]])
      assert Nx.shape(b) == {4, 1, 3}

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)
        assert Nx.shape(t) == {4, 3, 3}

        assert Nx.to_binary(t) ==
                 <<101::32-native, 102::32-native, 103::32-native, 201::32-native, 202::32-native,
                   203::32-native, 301::32-native, 302::32-native, 303::32-native, 104::32-native,
                   105::32-native, 106::32-native, 204::32-native, 205::32-native, 206::32-native,
                   304::32-native, 305::32-native, 306::32-native, 107::32-native, 108::32-native,
                   109::32-native, 207::32-native, 208::32-native, 209::32-native, 307::32-native,
                   308::32-native, 309::32-native, 110::32-native, 111::32-native, 112::32-native,
                   210::32-native, 211::32-native, 212::32-native, 310::32-native, 311::32-native,
                   312::32-native>>
      end)
    end

    test "{2, 1, 2} + {1, 2, 2, 1}" do
      a = Nx.tensor([[[[10], [20]], [[30], [40]]]])
      assert Nx.shape(a) == {1, 2, 2, 1}
      b = Nx.tensor([[[1, 2]], [[3, 4]]])
      assert Nx.shape(b) == {2, 1, 2}

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)
        assert Nx.shape(t) == {1, 2, 2, 2}

        assert Nx.to_binary(t) ==
                 <<11::32-native, 12::32-native, 21::32-native, 22::32-native, 33::32-native,
                   34::32-native, 43::32-native, 44::32-native>>
      end)
    end

    test "{2, 1, 1, 2} + {1, 2, 2, 1}" do
      a = Nx.tensor([[[[10], [20]], [[30], [40]]]])
      assert Nx.shape(a) == {1, 2, 2, 1}
      b = Nx.tensor([[[[1, 2]]], [[[3, 4]]]])
      assert Nx.shape(b) == {2, 1, 1, 2}

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)
        assert Nx.shape(t) == {2, 2, 2, 2}

        assert Nx.to_binary(t) ==
                 <<11::32-native, 12::32-native, 21::32-native, 22::32-native, 31::32-native,
                   32::32-native, 41::32-native, 42::32-native, 13::32-native, 14::32-native,
                   23::32-native, 24::32-native, 33::32-native, 34::32-native, 43::32-native,
                   44::32-native>>
      end)
    end

    test "{1, 2, 2} + {2, 1, 2}" do
      a = Nx.tensor([[[10], [20]], [[30], [40]]])
      b = Nx.tensor([[[1, 2]], [[3, 4]]])

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<11::32-native, 12::32-native, 21::32-native, 22::32-native, 33::32-native,
                   34::32-native, 43::32-native, 44::32-native>>

        assert Nx.shape(t) == {2, 2, 2}
      end)
    end

    test "raises when it cannot broadcast" do
      a = Nx.tensor([[1, 2], [3, 4]])
      b = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      assert_raise ArgumentError, ~r"cannot broadcast", fn -> Nx.add(a, b) end
    end
  end

  describe "convolution" do
    test "valid padding, no stride" do
      t1 = Nx.iota({2, 2, 4, 4}, type: {:f, 32})
      k1 = Nx.iota({1, 2, 2, 2}, type: {:f, 32})
      k2 = Nx.iota({8, 2, 4, 4}, type: {:f, 32})
      k3 = Nx.iota({2, 2, 3, 3}, type: {:f, 32})

      assert Nx.conv(t1, k1, strides: [1, 1], padding: :valid) ==
               Nx.tensor(
                 [
                   [
                     [
                       [440.0, 468.0, 496.0],
                       [552.0, 580.0, 608.0],
                       [664.0, 692.0, 720.0]
                     ]
                   ],
                   [
                     [
                       [1336.0, 1364.0, 1392.0],
                       [1448.0, 1476.0, 1504.0],
                       [1560.0, 1588.0, 1616.0]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )

      assert Nx.conv(t1, k2, strides: [1, 1], padding: :valid) ==
               Nx.tensor(
                 [
                   [
                     [[10416.0]],
                     [[26288.0]],
                     [[42160.0]],
                     [[58032.0]],
                     [[73904.0]],
                     [[89776.0]],
                     [[105_648.0]],
                     [[121_520.0]]
                   ],
                   [
                     [[26288.0]],
                     [[74928.0]],
                     [[123_568.0]],
                     [[172_208.0]],
                     [[220_848.0]],
                     [[269_488.0]],
                     [[318_128.0]],
                     [[366_768.0]]
                   ]
                 ],
                 type: {:f, 32}
               )

      assert Nx.conv(t1, k3, strides: [1, 1], padding: :valid) ==
               Nx.tensor(
                 [
                   [
                     [
                       [2793.0, 2946.0],
                       [3405.0, 3558.0]
                     ],
                     [
                       [7005.0, 7482.0],
                       [8913.0, 9390.0]
                     ]
                   ],
                   [
                     [
                       [7689.0, 7842.0],
                       [8301.0, 8454.0]
                     ],
                     [
                       [22269.0, 22746.0],
                       [24177.0, 24654.0]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end

    test "valid padding, stride" do
      t1 = Nx.iota({2, 2, 4, 4}, type: {:f, 32})
      k1 = Nx.iota({1, 2, 2, 2}, type: {:f, 32})

      assert Nx.conv(t1, k1, strides: [2, 1], padding: :valid) ==
               Nx.tensor(
                 [
                   [
                     [
                       [440.0, 468.0, 496.0],
                       [664.0, 692.0, 720.0]
                     ]
                   ],
                   [
                     [
                       [1336.0, 1364.0, 1392.0],
                       [1560.0, 1588.0, 1616.0]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end

    test "same padding, no stride" do
      t1 = Nx.iota({2, 2, 4, 4}, type: {:f, 32})
      k1 = Nx.iota({1, 2, 3, 3}, type: {:f, 32})

      assert Nx.conv(t1, k1, strides: [1, 1], padding: :same) ==
               Nx.tensor(
                 [
                   [
                     [
                       [1196.0, 1796.0, 1916.0, 1264.0],
                       [1881.0, 2793.0, 2946.0, 1923.0],
                       [2313.0, 3405.0, 3558.0, 2307.0],
                       [1424.0, 2072.0, 2156.0, 1380.0]
                     ]
                   ],
                   [
                     [
                       [3884.0, 5636.0, 5756.0, 3696.0],
                       [5337.0, 7689.0, 7842.0, 4995.0],
                       [5769.0, 8301.0, 8454.0, 5379.0],
                       [3344.0, 4760.0, 4844.0, 3044.0]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end

    test "3d conv" do
      t1 = Nx.iota({1, 3, 2, 2, 2}, type: {:f, 32})
      k1 = Nx.iota({3, 3, 2, 2, 2}, type: {:f, 32})

      assert Nx.conv(t1, k1, strides: [1, 1, 1], padding: :same) ==
               Nx.tensor(
                 [
                   [
                     [
                       [
                         [4324.0, 2156.0],
                         [2138.0, 1060.0]
                       ],
                       [
                         [2066.0, 1018.0],
                         [997.0, 488.0]
                       ]
                     ],
                     [
                       [
                         [10948.0, 5612.0],
                         [5738.0, 2932.0]
                       ],
                       [
                         [5954.0, 3034.0],
                         [3085.0, 1568.0]
                       ]
                     ],
                     [
                       [
                         [17572.0, 9068.0],
                         [9338.0, 4804.0]
                       ],
                       [
                         [9842.0, 5050.0],
                         [5173.0, 2648.0]
                       ]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end

    test "grouped convolutions" do
      lhs = Nx.iota({2, 4, 2, 2}, type: {:f, 32})
      rhs = Nx.iota({6, 2, 2, 2}, type: {:f, 32})

      assert Nx.conv(lhs, rhs, strides: 1, padding: :valid, feature_group_size: 2) ==
               Nx.tensor(
                 [
                   [[[140.0]], [[364.0]], [[588.0]], [[2572.0]], [[3308.0]], [[4044.0]]],
                   [[[588.0]], [[1836.0]], [[3084.0]], [[6092.0]], [[7852.0]], [[9612.0]]]
                 ],
                 type: {:f, 32}
               )
    end

    test "vectorization" do
      t =
        Nx.tensor([
          [
            [
              [0, 1, 2]
            ]
          ],
          [
            [
              [3, 4, 5]
            ]
          ]
        ])
        |> Nx.vectorize(x: 2)

      k =
        Nx.tensor([
          [
            [
              [2]
            ]
          ],
          [
            [
              [10]
            ]
          ]
        ])

      # same vectorized axis case
      assert Nx.conv(t, Nx.vectorize(k, x: 2)) ==
               Nx.tensor([
                 [
                   [
                     [0.0, 2.0, 4.0]
                   ]
                 ],
                 [
                   [
                     [30.0, 40.0, 50.0]
                   ]
                 ]
               ])
               |> Nx.vectorize(x: 2)

      # cross-product case
      assert Nx.conv(t, Nx.vectorize(Nx.reshape(k, {1, 2, 1, 1, 1}), z: 1, y: 2)) ==
               Nx.tensor([
                 [
                   [
                     [
                       [
                         [0.0, 2.0, 4.0]
                       ]
                     ],
                     [
                       [
                         [0.0, 10.0, 20.0]
                       ]
                     ]
                   ]
                 ],
                 [
                   [
                     [
                       [
                         [6.0, 8.0, 10.0]
                       ]
                     ],
                     [
                       [
                         [30.0, 40.0, 50.0]
                       ]
                     ]
                   ]
                 ]
               ])
               |> Nx.vectorize(x: 2, z: 1, y: 2)

      # vectorized tensor and devec kernel
      assert Nx.conv(t, k[0]) ==
               Nx.tensor([
                 [
                   [
                     [0.0, 2.0, 4.0]
                   ]
                 ],
                 [
                   [
                     [6.0, 8.0, 10.0]
                   ]
                 ]
               ])
               |> Nx.vectorize(x: 2)

      # devec tensor and vectorized kernel
      assert Nx.conv(Nx.devectorize(t)[0], Nx.vectorize(k, x: 2)) ==
               Nx.tensor([
                 [
                   [
                     [0.0, 2.0, 4.0]
                   ]
                 ],
                 [
                   [
                     [0.0, 10.0, 20.0]
                   ]
                 ]
               ])
               |> Nx.vectorize(x: 2)
    end

    test "vectorization - batch_group_size" do
      # get and assert the expected base shape for the output
      assert %{shape: {1, 3, 1, 1}} =
               Nx.conv(Nx.iota({3, 10, 1, 1}), Nx.iota({3, 10, 1, 1}), batch_group_size: 3)

      # using 2, 5 and 7 as vectorized sizes so that they have no common factor between each other
      # and between them and the batch_group_size: 3
      t = Nx.iota({3, 10, 1, 1}, vectorized_axes: [x: 2, y: 1, z: 7])
      k = Nx.iota({3, 10, 1, 1}, vectorized_axes: [x: 1, y: 5, z: 7])

      assert %{vectorized_axes: [x: 2, y: 5, z: 7], shape: {1, 3, 1, 1}} =
               Nx.conv(t, k, batch_group_size: 3)
    end
  end

  describe "pad" do
    test "with interior padding" do
      assert Nx.pad(Nx.tensor([[1, 2, 3], [4, 5, 6]]), 0, [{0, 0, 1}, {0, 0, 1}]) ==
               Nx.tensor([[1, 0, 2, 0, 3], [0, 0, 0, 0, 0], [4, 0, 5, 0, 6]])

      assert Nx.pad(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 2.0, [{-2, 1, 4}, {1, 3, 2}]) ==
               Nx.tensor([
                 [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                 [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                 [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                 [2.0, 4.0, 2.0, 2.0, 5.0, 2.0, 2.0, 6.0, 2.0, 2.0, 2.0],
                 [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
               ])
    end
  end

  describe "bitshift" do
    test "raises for negative numbers on the right" do
      assert_raise ArgumentError,
                   "cannot right shift by -1",
                   fn -> Nx.right_shift(Nx.tensor(1), -1) end
    end

    test "raises for negative numbers on the left" do
      assert_raise ArgumentError,
                   "cannot left shift by -1",
                   fn -> Nx.left_shift(Nx.tensor(1), -1) end
    end
  end

  describe "reverse" do
    test "works on all dims" do
      assert Nx.reverse(Nx.iota({10})) == Nx.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
      assert Nx.reverse(Nx.iota({2, 4})) == Nx.tensor([[7, 6, 5, 4], [3, 2, 1, 0]])

      assert Nx.reverse(Nx.iota({3, 3, 3})) ==
               Nx.tensor([
                 [[26, 25, 24], [23, 22, 21], [20, 19, 18]],
                 [[17, 16, 15], [14, 13, 12], [11, 10, 9]],
                 [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
               ])

      assert Nx.reverse(Nx.iota({2, 1, 4, 2})) ==
               Nx.tensor([
                 [[[15, 14], [13, 12], [11, 10], [9, 8]]],
                 [[[7, 6], [5, 4], [3, 2], [1, 0]]]
               ])
    end

    test "works with 1 dim" do
      assert Nx.reverse(Nx.iota({2, 3}), axes: [1]) == Nx.tensor([[2, 1, 0], [5, 4, 3]])
    end

    test "works with 2 dims" do
      assert Nx.reverse(Nx.iota({2, 3, 4}), axes: [0, 2]) ==
               Nx.tensor([
                 [
                   [15, 14, 13, 12],
                   [19, 18, 17, 16],
                   [23, 22, 21, 20]
                 ],
                 [
                   [3, 2, 1, 0],
                   [7, 6, 5, 4],
                   [11, 10, 9, 8]
                 ]
               ])
    end

    test "works with 3 dims" do
      assert Nx.reverse(Nx.iota({2, 2, 1, 3, 4}), axes: [1, 2, 4]) ==
               Nx.tensor([
                 [
                   [
                     [
                       [15, 14, 13, 12],
                       [19, 18, 17, 16],
                       [23, 22, 21, 20]
                     ]
                   ],
                   [
                     [
                       [3, 2, 1, 0],
                       [7, 6, 5, 4],
                       [11, 10, 9, 8]
                     ]
                   ]
                 ],
                 [
                   [
                     [
                       [39, 38, 37, 36],
                       [43, 42, 41, 40],
                       [47, 46, 45, 44]
                     ]
                   ],
                   [
                     [
                       [27, 26, 25, 24],
                       [31, 30, 29, 28],
                       [35, 34, 33, 32]
                     ]
                   ]
                 ]
               ])
    end

    test "works across types" do
      assert Nx.reverse(Nx.iota({2, 2, 3, 1, 2}, type: {:f, 32}), axes: [1, 2, 4]) ==
               Nx.tensor(
                 [
                   [
                     [
                       [[11.0, 10.0]],
                       [[9.0, 8.0]],
                       [[7.0, 6.0]]
                     ],
                     [
                       [[5.0, 4.0]],
                       [[3.0, 2.0]],
                       [[1.0, 0.0]]
                     ]
                   ],
                   [
                     [
                       [[23.0, 22.0]],
                       [[21.0, 20.0]],
                       [[19.0, 18.0]]
                     ],
                     [
                       [[17.0, 16.0]],
                       [[15.0, 14.0]],
                       [[13.0, 12.0]]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end
  end

  describe "inspect" do
    test "scalar" do
      assert inspect(Nx.tensor(123)) == """
             #Nx.Tensor<
               s32
               123
             >\
             """
    end

    test "n-dimensional" do
      assert inspect(Nx.tensor([[1, 2, 3], [4, 5, 6]])) == """
             #Nx.Tensor<
               s32[2][3]
               [
                 [1, 2, 3],
                 [4, 5, 6]
               ]
             >\
             """

      assert inspect(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) == """
             #Nx.Tensor<
               f32[2][3]
               [
                 [1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]
               ]
             >\
             """
    end

    test "limit" do
      assert inspect(Nx.tensor([1, 2]), limit: :infinity) == """
             #Nx.Tensor<
               s32[2]
               [1, 2]
             >\
             """

      assert inspect(Nx.tensor([[1, 2], [3, 4]]), limit: 3) == """
             #Nx.Tensor<
               s32[2][2]
               [
                 [1, 2],
                 [3, ...]
               ]
             >\
             """

      assert inspect(Nx.tensor([[1, 2], [3, 4], [5, 6]]), limit: 3) == """
             #Nx.Tensor<
               s32[3][2]
               [
                 [1, 2],
                 [3, ...],
                 ...
               ]
             >\
             """
    end

    test "infinity and nan for bf16" do
      bin = <<0xFF80::16-native, 0x7F80::16-native, 0xFFC1::16-native, 0xFF81::16-native>>

      assert inspect(Nx.from_binary(bin, {:bf, 16})) == """
             #Nx.Tensor<
               bf16[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end

    test "infinity and nan for f16" do
      bin = <<0xFC00::16-native, 0x7C00::16-native, 0xFC01::16-native, 0xFC0F::16-native>>

      assert inspect(Nx.from_binary(bin, {:f, 16})) == """
             #Nx.Tensor<
               f16[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end

    test "infinity and nan for f32" do
      bin =
        <<0xFF800000::32-native, 0x7F800000::32-native, 0xFF800001::32-native,
          0xFFC00001::32-native>>

      # Assert that none of them are indeed valid
      assert for(<<x::float-32-native <- bin>>, do: x) == []

      assert inspect(Nx.from_binary(bin, {:f, 32})) == """
             #Nx.Tensor<
               f32[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end

    test "more nan for f32" do
      bin =
        <<0x7F800001::32-native, 0xFF800002::32-native, 0x7FC00000::32-native,
          0xFFC00000::32-native>>

      # Assert that none of them are indeed valid
      assert for(<<x::float-32-native <- bin>>, do: x) == []

      assert inspect(Nx.from_binary(bin, {:f, 32})) == """
             #Nx.Tensor<
               f32[4]
               [NaN, NaN, NaN, NaN]
             >\
             """
    end

    test "infinity and nan for f64" do
      bin =
        <<0xFFF0000000000000::64-native, 0x7FF0000000000000::64-native,
          0x7FF0000000000001::64-native, 0x7FF8000000000001::64-native>>

      # Assert that none of them are indeed valid
      assert for(<<x::float-64-native <- bin>>, do: x) == []

      assert inspect(Nx.from_binary(bin, {:f, 64})) == """
             #Nx.Tensor<
               f64[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end

    test "more nan for f64" do
      bin =
        <<0xFFF0000000000001::64-native, 0x7FF0000000000002::64-native,
          0xFFF8000000000000::64-native, 0x7FF8000000000000::64-native>>

      # Assert that none of them are indeed valid
      assert for(<<x::float-64-native <- bin>>, do: x) == []

      assert inspect(Nx.from_binary(bin, {:f, 64})) == """
             #Nx.Tensor<
               f64[4]
               [NaN, NaN, NaN, NaN]
             >\
             """
    end

    test "cannot build an empty tensor" do
      assert_raise(ArgumentError, "cannot build an empty tensor", fn ->
        Nx.from_binary(<<>>, {:f, 64})
      end)
    end

    test "all dimensions named" do
      assert inspect(Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:batch, :x, :y])) ==
               """
               #Nx.Tensor<
                 s32[batch: 1][x: 3][y: 3]
                 [
                   [
                     [1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]
                   ]
                 ]
               >\
               """
    end

    test "some dimensions named" do
      assert inspect(Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:batch, nil, nil])) ==
               """
               #Nx.Tensor<
                 s32[batch: 1][3][3]
                 [
                   [
                     [1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]
                   ]
                 ]
               >\
               """
    end
  end

  describe "window aggregates" do
    test "computes a window sum" do
      assert Nx.window_sum(
               Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]]),
               {2, 1, 2},
               strides: [2, 1, 1],
               padding: [{2, 1}, {3, 1}, {1, 0}],
               window_dilations: [1, 2, 2]
             ) ==
               Nx.tensor([
                 [
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]
                 ],
                 [
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [4, 11, 14],
                   [10, 15, 19],
                   [0, 0, 0]
                 ]
               ])
    end

    test "computes a same padding window sum f64" do
      t = Nx.iota({3, 3}, type: {:f, 64})

      assert Nx.window_sum(t, {2, 1}, padding: :same) ==
               Nx.tensor([[3.0, 5.0, 7.0], [9.0, 11.0, 13.0], [6.0, 7.0, 8.0]], type: {:f, 64})
    end

    test "computes a same padding window product f64" do
      t = Nx.iota({3, 3}, type: {:f, 64})

      assert Nx.window_product(t, {2, 1}, padding: :same) ==
               Nx.tensor([[0.0, 4.0, 10.0], [18.0, 28.0, 40.0], [6.0, 7.0, 8.0]], type: {:f, 64})
    end

    test "computes a same padding window max f64" do
      t = Nx.iota({3, 3}, type: {:f, 64})

      assert Nx.window_max(t, {2, 1}, padding: :same) ==
               Nx.tensor([[3, 4, 5], [6, 7, 8], [6, 7, 8]], type: {:f, 64})
    end

    test "computes a same padding window min f64" do
      t = Nx.iota({3, 3}, type: {:f, 64})

      assert Nx.window_min(t, {2, 1}, padding: :same) ==
               Nx.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], type: {:f, 64})
    end
  end

  describe "put_slice" do
    test "with mixed types and sizes" do
      assert Nx.put_slice(Nx.tensor([0.0, 0.0]), [0], Nx.tensor([1])) == Nx.tensor([1.0, 0.0])
    end
  end

  describe "access" do
    test "supports an empty list" do
      tensor = Nx.tensor([1, 2, 3])
      assert tensor[[]] == tensor
    end

    test "supports an dynamic tensor" do
      tensor = Nx.tensor([1, 2, 3], names: [:row])
      assert tensor[Nx.tensor(1)] == Nx.tensor(2)
      assert tensor[[Nx.tensor(1)]] == Nx.tensor(2)
      assert tensor[[row: Nx.tensor(1)]] == Nx.tensor(2)
      assert tensor[Nx.tensor(-1)] == Nx.tensor(1)
    end

    test "raises on duplicated axis" do
      tensor = Nx.tensor([[1, 2, 3]], names: [:x, :y])

      assert_raise ArgumentError,
                   ~r"unknown or duplicate axis 1 found when slicing shape {1, 3}",
                   fn -> tensor[[y: 1, y: 2]] end
    end

    test "raises on invalid ranges" do
      tensor = Nx.tensor([[1, 2, 3]], names: [:x, :y])

      assert_raise ArgumentError,
                   ~r"slicing a tensor requires a non-empty range, got: 2..1//1",
                   fn -> tensor[[y: 2..1//1]] end
    end
  end

  describe "indexed_add" do
    test "property" do
      n = 5

      indices =
        Nx.tensor(
          for row <- 0..(n - 1), col <- 0..(n - 1) do
            [row, col]
          end
        )

      updates = List.duplicate(1, n * n) |> Nx.tensor()

      assert Nx.broadcast(1, {5, 5}) ==
               0 |> Nx.broadcast({5, 5}) |> Nx.indexed_add(indices, updates)

      indices =
        Nx.tensor(
          for row <- 0..(n - 1), col <- 0..(n - 1) do
            [0, row, col]
          end
        )

      assert Nx.broadcast(1, {1, 5, 5}) ==
               0 |> Nx.broadcast({1, 5, 5}) |> Nx.indexed_add(indices, updates)
    end

    test "single-index regression" do
      n = 5

      # 1D case
      zeros = List.duplicate(0, n)
      upd = Nx.tensor([1])

      for i <- 0..(n - 1) do
        result = Nx.tensor(List.replace_at(zeros, i, 1))

        assert result == Nx.indexed_add(Nx.tensor(zeros), Nx.tensor([[i]]), upd)
      end

      # 2D case
      zeros = List.duplicate(zeros, n)

      for i <- 0..(n - 1) do
        result =
          Nx.tensor(
            List.update_at(
              zeros,
              i,
              fn row ->
                List.update_at(row, i, fn _ -> 1 end)
              end
            )
          )

        assert result == Nx.indexed_add(Nx.tensor(zeros), Nx.tensor([[i, i]]), upd)
      end
    end

    test "raises when out of bounds" do
      t = Nx.tensor([[1, 2], [3, 4]])

      assert_raise ArgumentError, "index 3 is out of bounds for axis 0 in shape {2, 2}", fn ->
        Nx.indexed_add(t, Nx.tensor([[3, -10]]), Nx.tensor([1]))
      end

      assert_raise ArgumentError, "index -1 is out of bounds for axis 1 in shape {2, 2}", fn ->
        Nx.indexed_add(t, Nx.tensor([[0, -1]]), Nx.tensor([1]))
      end
    end

    test "complex regression" do
      # BinaryBackend used to break on Enum.sum over a complex-valued list
      assert Nx.tensor([1, 2], type: {:c, 64}) ==
               Nx.indexed_add(
                 Nx.broadcast(0, {2}),
                 Nx.tensor([[1], [0]]),
                 Nx.tensor([Complex.new(2), 1])
               )
    end

    test "vectorized target and non-vectorized index" do
      t = Nx.tensor([[0, 1, 2], [3, 4, 5]]) |> Nx.vectorize(x: 2)
      index = Nx.tensor([[0], [1], [1], [0]])
      updates = Nx.tensor([1, 2, 3, 4])

      assert Nx.indexed_add(t, index, updates) ==
               Nx.tensor([
                 [5, 6, 2],
                 [8, 9, 5]
               ])
               |> Nx.vectorize(x: 2)
    end

    test "non-vectorized target and vectorized index+updates" do
      t = Nx.tensor([0, 1, 2])

      assert %{shape: {2, 1}} =
               index =
               Nx.tensor([
                 [0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]
               ])
               |> Nx.reshape({2, 2, 2, 1})
               |> Nx.vectorize(x: 2, y: 2)

      updates =
        Nx.tensor([
          [1, 2],
          [3, 4]
        ])
        |> Nx.vectorize(x: 2)

      assert Nx.indexed_add(t, index, updates) ==
               Nx.tensor([
                 [
                   [3, 1, 2],
                   [1, 3, 2]
                 ],
                 [
                   [4, 4, 2],
                   [0, 8, 2]
                 ]
               ])
               |> Nx.vectorize(x: 2, y: 2)
    end
  end

  describe "indexed_put" do
    test "property" do
      n = 5

      indices =
        Nx.tensor(
          for row <- 0..(n - 1), col <- 0..(n - 1) do
            [row, col]
          end
        )

      updates = List.duplicate(1, n * n) |> Nx.tensor()

      assert Nx.broadcast(1, {5, 5}) ==
               0 |> Nx.broadcast({5, 5}) |> Nx.indexed_put(indices, updates)

      indices =
        Nx.tensor(
          for row <- 0..(n - 1), col <- 0..(n - 1) do
            [0, row, col]
          end
        )

      assert Nx.broadcast(1, {1, 5, 5}) ==
               0 |> Nx.broadcast({1, 5, 5}) |> Nx.indexed_put(indices, updates)
    end

    test "single-index regression" do
      n = 5

      # 1D case
      zeros = List.duplicate(0, n)
      upd = Nx.tensor([1])

      for i <- 0..(n - 1) do
        result = Nx.tensor(List.replace_at(zeros, i, 1))

        assert result == Nx.indexed_put(Nx.tensor(zeros), Nx.tensor([[i]]), upd)
      end

      # 2D case
      zeros = List.duplicate(zeros, n)

      for i <- 0..(n - 1) do
        result =
          Nx.tensor(
            List.update_at(
              zeros,
              i,
              fn row ->
                List.update_at(row, i, fn _ -> 1 end)
              end
            )
          )

        assert result == Nx.indexed_put(Nx.tensor(zeros), Nx.tensor([[i, i]]), upd)
      end
    end

    test "raises when out of bounds" do
      t = Nx.tensor([[1, 2], [3, 4]])

      assert_raise ArgumentError, "index 3 is out of bounds for axis 0 in shape {2, 2}", fn ->
        Nx.indexed_put(t, Nx.tensor([[3, -10]]), Nx.tensor([1]))
      end

      assert_raise ArgumentError, "index -1 is out of bounds for axis 1 in shape {2, 2}", fn ->
        Nx.indexed_put(t, Nx.tensor([[0, -1]]), Nx.tensor([1]))
      end
    end

    test "complex regression" do
      # BinaryBackend used to break on Enum.sum over a complex-valued list
      assert Nx.tensor([1, 2], type: {:c, 64}) ==
               Nx.indexed_put(
                 Nx.broadcast(0, {2}),
                 Nx.tensor([[1], [0]]),
                 Nx.tensor([Complex.new(2), 1])
               )
    end
  end

  describe "quotient/2" do
    test "raises for non-integer values" do
      msg =
        "quotient expects integer tensors as inputs and outputs an integer tensor, got: {:f, 32}"

      assert_raise ArgumentError, msg, fn ->
        Nx.quotient(10, 1.0)
      end

      assert_raise ArgumentError, msg, fn ->
        Nx.quotient(10.0, 2)
      end
    end
  end

  describe "reshape" do
    test "correctly adds names with same shape" do
      t = Nx.tensor([[1, 2], [3, 4]])

      assert Nx.reshape(t, {2, 2}, names: [:x, :y]) ==
               Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
    end
  end

  describe "flatten" do
    test "returns a flattened tensor given a N-Dimensional tensor" do
      t = Nx.iota({3, 3, 3})

      assert Nx.flatten(t) |> Nx.shape() == {Nx.size(t)}
    end

    test "returns a flattened tensor given N-Dimensional tensor with axes" do
      t = Nx.iota({1, 2, 3, 4, 5, 6})

      expected = Nx.iota({1, 2, 60, 6})

      assert expected == Nx.flatten(t, axes: [2, 3, 4])
    end

    test "returns tensor unchanged given a 1 Dimensional tensor" do
      t = Nx.iota({10})

      assert Nx.flatten(t) == t
    end
  end

  describe "broadcast" do
    test "correctly adds names with same shape" do
      t = Nx.tensor([[1, 2], [3, 4]])

      assert Nx.broadcast(t, {2, 2}, names: [:x, :y]) ==
               Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
    end
  end

  describe "tensor/2" do
    test "with non-finite" do
      assert Nx.tensor(:neg_infinity) == Nx.Constants.neg_infinity()
      assert Nx.tensor(:infinity) == Nx.Constants.infinity()
      assert Nx.tensor(:nan) == Nx.Constants.nan()
    end

    test "raises for empty list" do
      assert_raise(ArgumentError, "invalid value given to Nx.tensor/1, got: []", fn ->
        Nx.tensor([])
      end)
    end

    test "raises for non-numeric list" do
      assert_raise(ArgumentError, "invalid value given to Nx.tensor/1, got: :error", fn ->
        Nx.tensor([:error])
      end)
    end

    test "raises for nested lists with different shapes" do
      len3 = [1, 2, 3]
      len2 = [1, 2]

      assert_raise(ArgumentError, ~r/lists have different shapes/, fn ->
        Nx.tensor([len3, len2])
      end)
    end

    test "raises with tensor and different backends" do
      assert_raise ArgumentError,
                   "Nx.tensor/2 wants to allocate on backend ProcessBackend but it was given a tensor allocated on Nx.BinaryBackend",
                   fn ->
                     Nx.tensor(Nx.tensor([1, 2, 3], backend: Nx.BinaryBackend),
                       backend: ProcessBackend
                     )
                   end
    end
  end

  describe "from_binary/3" do
    test "raises for empty bitstring" do
      assert_raise(ArgumentError, "cannot build an empty tensor", fn ->
        Nx.from_binary("", {:u, 32})
      end)
    end
  end

  describe "to_batched/3" do
    test "works for all batch sizes less than or equal to {n, ...}" do
      for rows <- 1..10, cols <- 1..10, batch_size <- 1..rows do
        t = Nx.iota({rows, cols})

        batches = Nx.to_batched(t, batch_size, leftover: :discard)
        refute is_list(batches)
        assert Enum.count(batches) == div(rows, batch_size)

        batches = Nx.to_batched(t, batch_size, leftover: :repeat)
        refute is_list(batches)

        if rem(rows, batch_size) == 0 do
          assert Enum.count(batches) == div(rows, batch_size)
        else
          assert Enum.count(batches) == div(rows, batch_size) + 1
        end
      end
    end

    test "raises for scalars" do
      t = Nx.tensor(1)

      assert_raise(ArgumentError, ~r/cannot batch non-vectorized scalar tensor/, fn ->
        Nx.to_batched(t, 1)
      end)
    end
  end

  describe "compatible?/2" do
    test "non-composite types" do
      refute Nx.compatible?(Complex.new(0, 2), 2)
      refute Nx.compatible?(2, Complex.new(0, 2))
      refute Nx.compatible?(Complex.new(0, 2), ~VEC[2])
      refute Nx.compatible?(~VEC[2], Complex.new(0, 2))
      refute Nx.compatible?(2, ~VEC[2i])
      refute Nx.compatible?(~VEC[2i], 2)
      refute Nx.compatible?(~VEC[2], ~VEC[2i])
      assert Nx.compatible?(Complex.new(2), Complex.new(0, 2))
      assert Nx.compatible?(2, 0)
    end

    test "structs" do
      args = {1, Complex.new(1), Nx.tensor(1)}
      c = %Container{a: args, b: args}
      assert Nx.compatible?(c, c)
    end
  end

  describe "reshape/3" do
    test "raises when tensor shape and new shape are not compatible" do
      t = Nx.tensor([3, 3, 3])

      assert_raise(ArgumentError, ~r/not compatible with new shape/, fn ->
        Nx.reshape(t, {7})
      end)
    end
  end

  describe "new_axis/3" do
    test "raises when axis is out of range of the rank" do
      t = Nx.tensor([1, 2, 3])
      assert Nx.rank(t) == 1

      assert_raise(ArgumentError, ~r/must be a number between -2 and 1/, fn ->
        Nx.new_axis(t, 3)
      end)
    end

    test "the new axis can shift the existing axes to the left" do
      t = Nx.tensor([1, 2, 3], names: [:x])
      t = Nx.new_axis(t, 0, :batch)
      assert t.shape == {1, 3}
      assert t.names == [:batch, :x]
    end
  end

  describe "pad/3" do
    test "raises for non-scalar pad value" do
      assert_raise(ArgumentError, "padding value must be a scalar and non-vectorized", fn ->
        Nx.pad(Nx.tensor(1), Nx.tensor([0]), [])
      end)
    end
  end

  describe "shape/1" do
    test "raises for invalid, non-shape values" do
      assert_raise(ArgumentError, ~r/expected a shape/, fn ->
        Nx.shape("something else")
      end)
    end

    test "raises for invalid shapes" do
      assert_raise(ArgumentError, ~r/invalid dimension in axis 2/, fn ->
        Nx.shape({1, 2, :blep})
      end)
    end
  end

  describe "abs/1" do
    test "changes nothing for unsigned types" do
      t = Nx.tensor(10, type: {:u, 8})
      assert Nx.abs(t) == t

      t = Nx.tensor(10, type: {:u, 16})
      assert Nx.abs(t) == t

      t = Nx.tensor(10, type: {:u, 32})
      assert Nx.abs(t) == t

      t = Nx.tensor(10, type: {:u, 64})
      assert Nx.abs(t) == t
    end
  end

  describe "reduce_min/2" do
    test "works with non finites" do
      t =
        Nx.tensor([
          [:neg_infinity, :infinity, 1],
          [:neg_infinity, :nan, 1],
          [:neg_infinity, :neg_infinity, 1],
          [:neg_infinity, 0, 1],
          [:infinity, :infinity, 1],
          [:infinity, :nan, 1],
          [:infinity, :neg_infinity, 1],
          [:infinity, 0, 1],
          [:nan, :infinity, 1],
          [:nan, :nan, 1],
          [:nan, :neg_infinity, 1],
          [:nan, 0, 1]
        ])

      assert Nx.reduce_min(t, axes: [1]) ==
               Nx.tensor([
                 :neg_infinity,
                 :nan,
                 :neg_infinity,
                 :neg_infinity,
                 1,
                 :nan,
                 :neg_infinity,
                 0,
                 :nan,
                 :nan,
                 :nan,
                 :nan
               ])
    end

    test "removes all axes (scalar) when no axes are specified" do
      t = Nx.tensor([[1, 2, 3], [3, 1, 2]])
      out1 = Nx.reduce_min(t)
      assert out1 == Nx.tensor(1)
    end

    test "by default removes reduced axes" do
      t = Nx.tensor([[1, 2, 3], [3, 1, 2]])
      out1 = Nx.reduce_min(t, axes: [0])
      assert out1 == Nx.tensor([1, 1, 2])
      assert out1.shape == {3}
    end

    test "preserves and reduces given axes to size 1 with :keep_axes as true" do
      t = Nx.tensor([[1, 2, 3], [3, 1, 2]])
      assert t.shape == {2, 3}
      out1 = Nx.reduce_min(t, axes: [0], keep_axes: true)
      assert out1 == Nx.tensor([[1, 1, 2]])
      assert out1.shape == {1, 3}
    end

    test "preserves and reduces all dimensions to size 1 when no axes are specified with :keep_axes as true" do
      t = Nx.tensor([[1, 2, 3], [3, 1, 2]])
      assert t.shape == {2, 3}
      out1 = Nx.reduce_min(t, keep_axes: true)
      assert out1 == Nx.tensor([[1]])
      assert out1.shape == {1, 1}
    end
  end

  describe "reduce_max/2" do
    test "works with non finites" do
      t =
        Nx.tensor([
          [:neg_infinity, :infinity, 1],
          [:neg_infinity, :nan, 1],
          [:neg_infinity, :neg_infinity, 1],
          [:neg_infinity, 0, 1],
          [:infinity, :infinity, 1],
          [:infinity, :nan, 1],
          [:infinity, :neg_infinity, 1],
          [:infinity, 0, 1],
          [:nan, :infinity, 1],
          [:nan, :nan, 1],
          [:nan, :neg_infinity, 1],
          [:nan, 0, 1]
        ])

      assert Nx.reduce_max(t, axes: [1]) ==
               Nx.tensor([
                 :infinity,
                 :nan,
                 1,
                 1,
                 :infinity,
                 :nan,
                 :infinity,
                 :infinity,
                 :nan,
                 :nan,
                 :nan,
                 :nan
               ])
    end
  end

  describe "argmin/2" do
    test "works with non finites" do
      t =
        Nx.tensor([
          [:neg_infinity, :infinity, 1],
          [:neg_infinity, :nan, 1],
          [:neg_infinity, :neg_infinity, 1],
          [:neg_infinity, 0, 1],
          [:infinity, :infinity, 1],
          [:infinity, :nan, 1],
          [:infinity, :neg_infinity, 1],
          [:infinity, 0, 1],
          [:nan, :infinity, 1],
          [:nan, :nan, 1],
          [:nan, :neg_infinity, 1],
          [:nan, 0, 1]
        ])

      assert Nx.argmin(t, axis: 1) == Nx.tensor([0, 1, 0, 0, 2, 1, 1, 1, 0, 1, 0, 0])
    end

    test "raises for invalid :tie_break option" do
      assert_raise(
        ArgumentError,
        "unknown value for :tie_break, expected :high or :low, got: :blep",
        fn ->
          Nx.argmin(Nx.tensor(1), tie_break: :blep)
        end
      )
    end
  end

  describe "argmax" do
    test "works with non finites" do
      t =
        Nx.tensor([
          [:neg_infinity, :infinity, 1],
          [:neg_infinity, :nan, 1],
          [:neg_infinity, :neg_infinity, 1],
          [:neg_infinity, 0, 1],
          [:infinity, :infinity, 1],
          [:infinity, :nan, 1],
          [:infinity, :neg_infinity, 1],
          [:infinity, 0, 1],
          [:nan, :infinity, 1],
          [:nan, :nan, 1],
          [:nan, :neg_infinity, 1],
          [:nan, 0, 1]
        ])

      assert Nx.argmax(t, axis: 1) == Nx.tensor([1, 1, 2, 2, 0, 1, 0, 0, 0, 1, 0, 0])
    end
  end

  describe "aggregate_window_op" do
    test "option :window_dilations can be an integer" do
      t = Nx.tensor([1, 2, 3, 4, 5, 6, 7])
      out = Nx.window_max(t, {2}, window_dilations: 4)
      assert out == Nx.tensor([5, 6, 7])
    end

    test "option :padding can be :same" do
      t = Nx.tensor([1, 2, 3])
      out = Nx.window_max(t, {2}, padding: :same)
      assert out == Nx.tensor([2, 3, 3])
    end

    test "raises for invalid :padding option" do
      t = Nx.tensor([1, 2, 3])

      assert_raise(
        ArgumentError,
        ~r/invalid padding mode specified/,
        fn ->
          Nx.window_max(t, {2}, padding: :whatever)
        end
      )
    end
  end

  describe "reduce/4" do
    test "reduces dimension to 1 but does not remove it with :keep_axes as true" do
      t = Nx.tensor([1, 2, 3])
      out = Nx.reduce(t, 0, [keep_axes: true], fn x, y -> Nx.add(x, y) end)
      assert Nx.shape(out) == {1}
      assert out == Nx.tensor([6])
    end
  end

  describe "window_reduce/5" do
    test "works with :window_dilations option as an integer" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      opts = [window_dilations: 2]
      out = Nx.window_reduce(t, 0, {2, 2}, opts, fn x, acc -> Nx.max(x, acc) end)
      assert out == Nx.tensor([[9]])
    end

    test "works with :strides option as an integer" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      opts = [strides: 1]
      out = Nx.window_reduce(t, 0, {2, 2}, opts, fn x, acc -> Nx.max(x, acc) end)
      assert out == Nx.tensor([[5, 6], [8, 9]])
    end

    test "works with :padding option as a list of shape-matching integer tuples" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      opts = [padding: [{0, 0}, {0, 1}]]
      out = Nx.window_reduce(t, 0, {2, 2}, opts, fn x, acc -> Nx.max(x, acc) end)
      assert Nx.shape(out) == {2, 3}

      assert out ==
               Nx.tensor([
                 [5, 6, 6],
                 [8, 9, 9]
               ])
    end

    test "raises for invalid :padding option" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      opts = [padding: :whatever]

      assert_raise(
        ArgumentError,
        ~r/invalid padding mode specified/,
        fn ->
          Nx.window_reduce(t, 0, {2, 2}, opts, fn x, acc -> max(x, acc) end)
        end
      )
    end
  end

  describe "dot/2" do
    test "second arg call be a scalar" do
      t = Nx.tensor([[1, 2, 3]])
      assert Nx.shape(t) == {1, 3}
      out = Nx.dot(t, 3)
      assert Nx.shape(out) == {1, 3}
      assert out == Nx.tensor([[3, 6, 9]])
    end
  end

  describe "dot/4" do
    test "does not re-sort the contracting axes" do
      left = Nx.iota({2, 7, 8, 3, 1})
      right = Nx.iota({1, 8, 3, 7, 3})

      result = Nx.dot(left, [3, 1, 2], right, [2, 3, 1])

      assert {2, 1, 1, 3} == result.shape

      # Expected result obtained from pytorch
      assert result ==
               Nx.tensor([
                 [
                   [
                     [3_731_448, 3_745_476, 3_759_504]
                   ]
                 ],
                 [
                   [
                     [10_801_560, 10_843_812, 10_886_064]
                   ]
                 ]
               ])
    end
  end

  describe "dot/6" do
    test "works with batched dot and different size non-batch dims" do
      t1 = Nx.iota({3, 2, 4, 1})
      t2 = Nx.iota({3, 4, 2, 2})

      assert Nx.dot(t1, [1, 2], [0], t2, [2, 1], [0]) ==
               Nx.tensor([[[252, 280]], [[2172, 2264]], [[6140, 6296]]])
    end

    test "works with multiple batch dimensions" do
      t1 = Nx.iota({3, 2, 1, 3, 2})
      t2 = Nx.iota({3, 2, 1, 1, 4})

      assert Nx.dot(t1, [2], [0, 1], t2, [3], [0, 1]) ==
               Nx.tensor([
                 [
                   [
                     [[[0, 0, 0, 0]], [[0, 1, 2, 3]]],
                     [[[0, 2, 4, 6]], [[0, 3, 6, 9]]],
                     [[[0, 4, 8, 12]], [[0, 5, 10, 15]]]
                   ],
                   [
                     [[[24, 30, 36, 42]], [[28, 35, 42, 49]]],
                     [[[32, 40, 48, 56]], [[36, 45, 54, 63]]],
                     [[[40, 50, 60, 70]], [[44, 55, 66, 77]]]
                   ]
                 ],
                 [
                   [
                     [[[96, 108, 120, 132]], [[104, 117, 130, 143]]],
                     [[[112, 126, 140, 154]], [[120, 135, 150, 165]]],
                     [[[128, 144, 160, 176]], [[136, 153, 170, 187]]]
                   ],
                   [
                     [[[216, 234, 252, 270]], [[228, 247, 266, 285]]],
                     [[[240, 260, 280, 300]], [[252, 273, 294, 315]]],
                     [[[264, 286, 308, 330]], [[276, 299, 322, 345]]]
                   ]
                 ],
                 [
                   [
                     [[[384, 408, 432, 456]], [[400, 425, 450, 475]]],
                     [[[416, 442, 468, 494]], [[432, 459, 486, 513]]],
                     [[[448, 476, 504, 532]], [[464, 493, 522, 551]]]
                   ],
                   [
                     [[[600, 630, 660, 690]], [[620, 651, 682, 713]]],
                     [[[640, 672, 704, 736]], [[660, 693, 726, 759]]],
                     [[[680, 714, 748, 782]], [[700, 735, 770, 805]]]
                   ]
                 ]
               ])
    end

    test "works with batch dimension of size 1" do
      t = Nx.tensor([[[1, 2, 3], [4, 5, 6]]])
      assert Nx.dot(t, [2], [0], t, [2], [0]) == Nx.tensor([[[14, 32], [32, 77]]])
    end

    test "vectorization regression" do
      l = Nx.tensor([[[[0, 1, 2], [0, -1, -2]]]]) |> Nx.tile([5, 1, 1, 1]) |> Nx.vectorize(x: 5)

      assert l.shape == {1, 2, 3}

      r =
        Nx.tensor([
          [1, 1, 1, 1],
          [2, 2, 2, 2],
          [3, 3, 3, 3]
        ])

      assert Nx.dot(l, r) ==
               Nx.tensor([
                 [
                   [8, 8, 8, 8],
                   [-8, -8, -8, -8]
                 ],
                 [
                   [8, 8, 8, 8],
                   [-8, -8, -8, -8]
                 ],
                 [
                   [8, 8, 8, 8],
                   [-8, -8, -8, -8]
                 ],
                 [
                   [8, 8, 8, 8],
                   [-8, -8, -8, -8]
                 ],
                 [
                   [8, 8, 8, 8],
                   [-8, -8, -8, -8]
                 ]
               ])
               |> Nx.vectorize(x: 5)
               |> Nx.reshape({1, 2, 4})
    end
  end

  describe "reverse/2" do
    test "does nothing when tensor is scalar" do
      t = Nx.tensor(1)
      assert Nx.reverse(t) == t
    end
  end

  describe "conv/3" do
    test "raises when input rank is below 3" do
      kernel = Nx.broadcast(Nx.tensor(0), {2, 2, 2})
      t = Nx.tensor([[1, 2]])
      assert Nx.rank(t) == 2

      assert_raise(ArgumentError, ~r/input shape in conv requires at least rank 3/, fn ->
        Nx.conv(t, kernel)
      end)
    end

    test "raises when kernel rank is below 3" do
      kernel = Nx.broadcast(Nx.tensor(0), {2, 2})
      assert Nx.rank(kernel) == 2
      t = Nx.iota({4, 4, 4})

      assert_raise(ArgumentError, ~r/kernel shape in conv requires at least rank 3/, fn ->
        Nx.conv(t, kernel)
      end)
    end

    test "raises when input channels are mismatched" do
      kernel = Nx.broadcast(Nx.tensor(0), {2, 2, 2})
      t = Nx.iota({4, 4, 4})

      assert_raise(
        ArgumentError,
        ~r/size of input channels divided by feature groups must match size of kernel channels/,
        fn ->
          Nx.conv(t, kernel)
        end
      )
    end

    test "raises when :feature_group_size cannot divide evenly into the input channels of the kernel" do
      t = Nx.iota({3, 2, 2})
      kernel = Nx.broadcast(Nx.tensor(1.0), {3, 1, 1})

      assert_raise(
        ArgumentError,
        ~r/size of kernel output channels must be evenly divisible by feature groups/,
        fn ->
          Nx.conv(t, kernel, feature_group_size: 2)
        end
      )
    end

    test "raises if both :feature_group_size and :batch_group_size are greater than 1" do
      t = Nx.iota({3, 2, 2})
      kernel = Nx.broadcast(Nx.tensor(1.0), {1, 1, 1})

      assert_raise(
        ArgumentError,
        ~r/either batch groups or feature groups must be 1/,
        fn ->
          Nx.conv(t, kernel, feature_group_size: 2, batch_group_size: 2)
        end
      )
    end

    test "raises if :batch_group_size does not evenly divide batch size" do
      t = Nx.iota({3, 2, 2})
      kernel = Nx.broadcast(Nx.tensor(1.0), {1, 1, 1})

      assert_raise(
        ArgumentError,
        ~r/batch groups must evenly divide input batch size/,
        fn ->
          Nx.conv(t, kernel, batch_group_size: 2)
        end
      )
    end

    test "raises if :batch_group_size is not a multiple of output feature channels" do
      t = Nx.iota({3, 2, 3, 3})
      kernel = Nx.iota({8, 2, 2, 2})

      assert_raise(
        ArgumentError,
        ~r/size of kernel output channels must be evenly divisible by batch groups/,
        fn ->
          Nx.conv(t, kernel, batch_group_size: 3)
        end
      )
    end

    test "raises when :strides length does not match spatial dims (input shape without 1st two dims)" do
      kernel = Nx.broadcast(Nx.tensor(0), {2, 2, 2})
      t = Nx.iota({4, 2, 4})

      assert_raise(ArgumentError, ~r/rank of strides much match rank of spatial dimension/, fn ->
        Nx.conv(t, kernel, strides: [1, 1, 1, 1])
      end)
    end

    defp conv_raise_for_options(message, opts) do
      kernel = Nx.broadcast(Nx.tensor(0), {2, 2, 2})
      t = Nx.iota({4, 2, 4})

      assert_raise(ArgumentError, message, fn ->
        Nx.conv(t, kernel, opts)
      end)
    end

    test "raises when :input_dilation is not positive" do
      message = ~r/input dilation of each dimension must be a positive integer/
      conv_raise_for_options(message, input_dilation: 0)
    end

    test "works when :input_dilation is valid length list with valid positive integers" do
      kernel = Nx.broadcast(Nx.tensor(1.0), {1, 2, 1})
      t = Nx.iota({2, 2, 2})
      out = Nx.conv(t, kernel, input_dilation: [1])
      assert Nx.shape(out) == {2, 1, 2}
      assert out == Nx.tensor([[[2.0, 4.0]], [[10.0, 12.0]]])
    end

    test "raises when :input_dilation list is not the same length as spatial dims (inputs shape without 1st two dims)" do
      message = ~r/must specify dilation for each spatial dimension of the input/
      conv_raise_for_options(message, input_dilation: [1, 1, 1])
    end

    test "raises when :input_dilation list has a non-positive int" do
      message = "input dilation of each dimension must be a positive integer, got [-1]"
      conv_raise_for_options(message, input_dilation: [-1])
    end

    test "raises when :input_dilation list has a non-int" do
      message = "input dilation of each dimension must be a positive integer, got [1.0]"
      conv_raise_for_options(message, input_dilation: [1.0])
    end

    test "raises when :input_dilation is invalid type" do
      message =
        "input dilation must be a positive integer or list of positive integers, got {1.0}"

      conv_raise_for_options(message, input_dilation: {1.0})
    end

    # kernel dilation
    test "raises when :kernel_dilation is not positive" do
      message = ~r/kernel dilation of each dimension must be a positive integer/
      conv_raise_for_options(message, kernel_dilation: 0)
    end

    test "raises when :kernel_dilation list is not the same length as spatial dims (inputs shape without 1st two dims)" do
      message = ~r/must specify dilation for each spatial dimension of the kernel/
      conv_raise_for_options(message, kernel_dilation: [1, 1, 1])
    end

    test "raises when :kernel_dilation list has a non-positive int" do
      message = "kernel dilation of each dimension must be a positive integer, got [-1]"
      conv_raise_for_options(message, kernel_dilation: [-1])
    end

    test "raises when :kernel_dilation list has a non-int" do
      message = "kernel dilation of each dimension must be a positive integer, got [1.0]"
      conv_raise_for_options(message, kernel_dilation: [1.0])
    end

    test "raises when :kernel_dilation is invalid type" do
      message =
        "kernel dilation must be a positive integer or list of positive integers, got {1.0}"

      conv_raise_for_options(message, kernel_dilation: {1.0})
    end

    test "works when :padding is a valid config-list" do
      kernel = Nx.broadcast(Nx.tensor(1.0), {1, 2, 1})
      t = Nx.iota({2, 2, 2})
      out = Nx.conv(t, kernel, padding: [{0, 0}])
      assert Nx.shape(out) == {2, 1, 2}
      assert out == Nx.tensor([[[2.0, 4.0]], [[10.0, 12.0]]])
    end

    test "raises when :padding is an invalid type" do
      message = ~r/invalid padding/
      conv_raise_for_options(message, padding: :bad_value)
    end

    test "supports complex values" do
      # These are basically the doctest but with combinations
      # where either the tensor or the kernel (or both)
      # are complex-typed

      types = [f: 32, c: 64]

      for tensor_type <- types,
          kernel_type <- types,
          tensor_type != {:f, 32} or kernel_type != {:f, 32} do
        # first doctest
        tensor = Nx.iota({1, 1, 3, 3}, type: tensor_type)
        kernel = Nx.iota({4, 1, 1, 1}, type: kernel_type)

        expected_output =
          Nx.tensor(
            [
              [
                [
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]
                ],
                [
                  [0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]
                ],
                [
                  [0.0, 2.0, 4.0],
                  [6.0, 8.0, 10.0],
                  [12.0, 14.0, 16.0]
                ],
                [
                  [0.0, 3.0, 6.0],
                  [9.0, 12.0, 15.0],
                  [18.0, 21.0, 24.0]
                ]
              ]
            ],
            type: Nx.Type.merge(tensor_type, kernel_type)
          )

        assert_all_close(expected_output, Nx.conv(tensor, kernel, strides: [1, 1]))

        # second doctest
        tensor = Nx.iota({1, 1, 3, 3}, type: tensor_type)
        kernel = Nx.iota({4, 1, 2, 1}, type: kernel_type)

        expected_output =
          Nx.tensor(
            [
              [
                [
                  [3.0, 5.0],
                  [0.0, 0.0]
                ],
                [
                  [9.0, 15.0],
                  [6.0, 10.0]
                ],
                [
                  [15.0, 25.0],
                  [12.0, 20.0]
                ],
                [
                  [21.0, 35.0],
                  [18.0, 30.0]
                ]
              ]
            ],
            type: Nx.Type.merge(tensor_type, kernel_type)
          )

        assert_all_close(
          expected_output,
          Nx.conv(tensor, kernel, strides: 2, padding: :same, kernel_dilation: [2, 1])
        )
      end
    end
  end

  describe "clip/3" do
    test "raises when min arg is non-scalar" do
      t = Nx.iota({4})
      min = Nx.iota({2})
      max = 3

      assert_raise(
        ArgumentError,
        "min value must be a non-vectorized scalar shape, got shape {2} and vectorized axes []",
        fn ->
          Nx.clip(t, min, max)
        end
      )
    end

    test "raises when max arg is non-scalar" do
      t = Nx.iota({4})
      min = Nx.iota({})
      max = Nx.iota({3})

      assert_raise(
        ArgumentError,
        "max value must be a non-vectorized scalar shape, got shape {3} and vectorized axes []",
        fn ->
          Nx.clip(t, min, max)
        end
      )
    end

    test "works with infinities" do
      t = Nx.tensor([1.0, 2.0, 3.0])
      assert t == Nx.clip(t, Nx.Constants.neg_infinity(), Nx.Constants.infinity())
    end
  end

  describe "concatenate/2" do
    test "middle axis" do
      t1 = Nx.iota({2, 2, 2})
      t2 = Nx.add(t1, 10)
      t3 = Nx.add(t1, 20)

      assert Nx.concatenate([t1, t2, t3], axis: 1) ==
               Nx.tensor([
                 [
                   [0, 1],
                   [2, 3],
                   [10, 11],
                   [12, 13],
                   [20, 21],
                   [22, 23]
                 ],
                 [
                   [4, 5],
                   [6, 7],
                   [14, 15],
                   [16, 17],
                   [24, 25],
                   [26, 27]
                 ]
               ])
    end

    test "raises for an empty list of tensors" do
      assert_raise(ArgumentError, "no tensors were given to concatenate", fn ->
        Nx.concatenate([])
      end)
    end
  end

  describe "sort/1" do
    test "works" do
      t = Nx.tensor([3, 2, 1, 0])
      assert Nx.sort(t) == Nx.tensor([0, 1, 2, 3])
    end

    test "works with non-symmetric tensors" do
      tensor =
        Nx.tensor(
          [
            [
              [4, 5],
              [2, 5],
              [5, 0]
            ],
            [
              [1, 9],
              [2, 1],
              [2, 1]
            ],
            [
              [0, -1],
              [-1, 0],
              [0, -1]
            ],
            [
              [-1, 0],
              [0, -1],
              [-1, 0]
            ]
          ],
          names: [:x, :y, :z]
        )

      assert Nx.sort(tensor, axis: :x) ==
               Nx.tensor(
                 [
                   [
                     [-1, -1],
                     [-1, -1],
                     [-1, -1]
                   ],
                   [
                     [0, 0],
                     [0, 0],
                     [0, 0]
                   ],
                   [
                     [1, 5],
                     [2, 1],
                     [2, 0]
                   ],
                   [
                     [4, 9],
                     [2, 5],
                     [5, 1]
                   ]
                 ],
                 names: [:x, :y, :z]
               )
    end
  end

  describe "sort/2" do
    test "raises for invalid direction" do
      t = Nx.tensor([3, 2, 1, 0])

      assert_raise(
        ArgumentError,
        "unknown value for :direction, expected :asc or :desc, got: :invalid",
        fn ->
          Nx.sort(t, direction: :invalid)
        end
      )
    end
  end

  describe "argsort/2" do
    test "works with non-symmetric tensors" do
      tensor =
        Nx.tensor(
          [
            [[4, 5], [2, 5], [5, 0]],
            [[1, 9], [2, 1], [2, 1]],
            [[0, -1], [-1, 0], [0, -1]],
            [[-1, 0], [0, -1], [-1, 0]]
          ],
          names: [:x, :y, :z]
        )

      assert Nx.argsort(tensor, axis: :x) ==
               Nx.tensor(
                 [
                   [
                     [3, 2],
                     [2, 3],
                     [3, 2]
                   ],
                   [
                     [2, 3],
                     [3, 2],
                     [2, 0]
                   ],
                   [
                     [1, 0],
                     [0, 1],
                     [1, 3]
                   ],
                   [
                     [0, 1],
                     [1, 0],
                     [0, 1]
                   ]
                 ],
                 names: [:x, :y, :z]
               )
    end

    test "raises for invalid direction" do
      t = Nx.tensor([3, 2, 1, 0])

      assert_raise(
        ArgumentError,
        "unknown value for :direction, expected :asc or :desc, got: :invalid",
        fn ->
          Nx.argsort(t, direction: :invalid)
        end
      )
    end
  end

  describe "take" do
    test "accepts scalar and number as index" do
      assert Nx.take(Nx.iota({4}), 2) == Nx.tensor(2)
      assert Nx.take(Nx.iota({4}), Nx.tensor(2)) == Nx.tensor(2)
      assert Nx.take(Nx.iota({4}), Nx.tensor([0, 2])) == Nx.tensor([0, 2])
    end
  end

  describe "take_diagonal/2" do
    test "extracts valid diagonal given no offset" do
      diag =
        {3, 3}
        |> Nx.iota()
        |> Nx.take_diagonal()

      assert diag == Nx.tensor([0, 4, 8])
    end

    test "extracts valid diagonal when breadth is greater than length" do
      diag =
        {3, 4}
        |> Nx.iota()
        |> Nx.take_diagonal()

      assert diag == Nx.tensor([0, 5, 10])
    end

    test "extracts valid diagonal when length is greater than breadth" do
      diag =
        {4, 3}
        |> Nx.iota()
        |> Nx.take_diagonal()

      assert diag == Nx.tensor([0, 4, 8])
    end

    test "extracts valid diagonal given positive offset" do
      diag =
        {3, 3}
        |> Nx.iota()
        |> Nx.take_diagonal(offset: 1)

      assert diag == Nx.tensor([1, 5])
    end

    test "extracts valid diagonal given negative offset" do
      diag =
        {3, 3}
        |> Nx.iota()
        |> Nx.take_diagonal(offset: -1)

      assert diag == Nx.tensor([3, 7])
    end

    test "extracts valid diagonal given batched matrices" do
      diag =
        {2, 3, 3}
        |> Nx.iota()
        |> Nx.take_diagonal()

      assert diag == Nx.tensor([[0, 4, 8], [9, 13, 17]])
    end

    test "raises error given tensor with invalid rank" do
      t = Nx.iota({3})

      assert_raise(
        ArgumentError,
        "take_diagonal/2 expects tensor of rank 2 or higher, got tensor of rank: 1",
        fn -> Nx.take_diagonal(t) end
      )
    end

    test "raises error given invalid positive offset" do
      t = Nx.iota({3, 3})

      assert_raise(
        ArgumentError,
        "offset must be less than length of axis 1 when positive, got: 4",
        fn -> Nx.take_diagonal(t, offset: 4) end
      )
    end

    test "raisese error given invalid negative offset" do
      t = Nx.iota({3, 3})

      assert_raise(
        ArgumentError,
        "absolute value of offset must be less than length of axis 0 when negative, got: -3",
        fn -> Nx.take_diagonal(t, offset: -3) end
      )
    end
  end

  describe "make_diagonal/2" do
    test "constructs valid diagonal given no offset" do
      diag =
        [1, 2, 3]
        |> Nx.tensor()
        |> Nx.make_diagonal()

      assert diag == Nx.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    end

    test "constructs valid diagonal given positive offset" do
      diag =
        [1, 2, 3]
        |> Nx.tensor()
        |> Nx.make_diagonal(offset: 1)

      assert diag == Nx.tensor([[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3], [0, 0, 0, 0]])
    end

    test "constructs valid diagonal given negative offset" do
      diag =
        [1, 2, 3]
        |> Nx.tensor()
        |> Nx.make_diagonal(offset: -1)

      assert diag == Nx.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]])
    end

    test "raises error given tensor with invalid rank" do
      t = Nx.iota({3, 3, 3})

      assert_raise(
        ArgumentError,
        "make_diagonal/2 expects tensor of rank 1, got tensor of rank: 3",
        fn -> Nx.make_diagonal(t) end
      )
    end
  end

  describe "put_diagonal/2" do
    test "puts valid diagonal given no offset" do
      target = Nx.iota({3, 3})

      diag =
        [1, 2, 3]
        |> Nx.tensor()
        |> then(&Nx.put_diagonal(target, &1))

      assert diag == Nx.tensor([[1, 1, 2], [3, 2, 5], [6, 7, 3]])
    end

    test "puts valid diagonal given positive offset" do
      target = Nx.broadcast(0, {4, 4})

      diag =
        [1, 2, 3]
        |> Nx.tensor()
        |> then(&Nx.put_diagonal(target, &1, offset: 1))

      assert diag == Nx.tensor([[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3], [0, 0, 0, 0]])
    end

    test "puts valid diagonal given negative offset" do
      target = Nx.broadcast(0, {4, 4})

      diag =
        [1, 2, 3]
        |> Nx.tensor()
        |> then(&Nx.put_diagonal(target, &1, offset: -1))

      assert diag == Nx.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]])
    end

    test "raises error given tensor with invalid rank" do
      t = Nx.iota({3, 3, 3})
      u = Nx.iota({3})

      assert_raise(
        ArgumentError,
        "put_diagonal/3 expects tensor of rank 2, got tensor of rank: 3",
        fn -> Nx.put_diagonal(t, u) end
      )
    end

    test "raises error given diagonal with invalid rank" do
      t = Nx.iota({3, 3})
      u = Nx.iota({3, 3})

      assert_raise(
        ArgumentError,
        "put_diagonal/3 expects diagonal of rank 1, got tensor of rank: 2",
        fn -> Nx.put_diagonal(t, u) end
      )
    end

    test "raises error given diagonal tensor with invalid length, given no offset" do
      t = Nx.iota({3, 3})
      u = Nx.iota({2})

      assert_raise(
        ArgumentError,
        "expected diagonal tensor of length: 3, got diagonal tensor of length: 2",
        fn -> Nx.put_diagonal(t, u) end
      )
    end

    test "raises error given diagonal tensor with invalid length, given a valid offset" do
      t = Nx.iota({3, 3})
      u = Nx.iota({3})

      assert_raise(
        ArgumentError,
        "expected diagonal tensor of length: 2, got diagonal tensor of length: 3",
        fn -> Nx.put_diagonal(t, u, offset: 1) end
      )
    end

    test "raises error given an invalid positive offset" do
      t = Nx.iota({3, 3})
      u = Nx.iota({3})

      assert_raise(
        ArgumentError,
        "offset must be less than length of axis 1 when positive, got: 4",
        fn -> Nx.put_diagonal(t, u, offset: 4) end
      )
    end

    test "raises error given an invalid negative offset" do
      t = Nx.iota({3, 3})
      u = Nx.iota({3})

      assert_raise(
        ArgumentError,
        "absolute value of offset must be less than length of axis 0 when negative, got: -3",
        fn -> Nx.put_diagonal(t, u, offset: -3) end
      )
    end
  end

  describe "indexed_add/3" do
    test "can emulate take_along_axis" do
      # One can also convert the indices produced by argsort into suitable
      # indices for indexed_add as below.
      # The following example emulates `take_along_axis/3` with `indexed_add/3`

      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])
      axis = 1
      i = Nx.argsort(t, axis: axis, direction: :desc)

      assert Nx.tensor([
               [1, 0],
               [1, 0],
               [0, 1]
             ]) == i

      num_elements = t |> Nx.shape() |> Tuple.product()

      iotas =
        Enum.map(0..(Nx.rank(t) - 1)//1, fn axis ->
          t |> Nx.shape() |> Nx.iota(axis: axis) |> Nx.reshape({num_elements, 1})
        end)

      iotas = List.replace_at(iotas, axis, Nx.reshape(i, {num_elements, 1}))

      indices = Nx.concatenate(iotas, axis: 1)

      assert Nx.tensor([[5, 4], [3, 2], [1, 0]]) ==
               Nx.indexed_add(
                 Nx.broadcast(0, Nx.shape(t)),
                 indices,
                 Nx.reshape(t, {num_elements})
               )
    end

    test "works with all integer types in indices" do
      for kind <- [:u, :s], width <- [2, 4, 8, 16, 32, 64] do
        indices = Nx.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], type: {kind, width})

        assert Nx.add(Nx.iota({2, 2}), 1) ==
                 Nx.indexed_add(Nx.broadcast(0, {2, 2}), indices, Nx.tensor([1, 2, 3, 4]))
      end
    end

    test "fails for all non-integer types in indices" do
      for type <- [f: 16, bf: 16, f: 32, f: 64, c: 64, c: 128] do
        indices = Nx.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], type: type)

        assert_raise ArgumentError,
                     "indices must be an integer tensor, got type: #{inspect(type)}",
                     fn ->
                       Nx.indexed_add(Nx.broadcast(0, {2, 2}), indices, Nx.tensor([1, 2, 3, 4]))
                     end
      end
    end
  end

  describe "serialize/deserialize" do
    test "v1 regression" do
      serialized = [
        <<131, 104, 3, 97, 1, 100, 0, 6, 108, 105, 116, 116, 108, 101, 104, 3, 100, 0, 29, 69,
          108, 105, 120, 105, 114, 46, 78, 120, 46, 67, 111, 110, 116, 97, 105, 110, 101, 114, 46,
          67, 111, 110, 116, 97, 105, 110, 101, 114, 108, 0, 0, 0, 2, 104, 2, 100, 0, 1, 97, 104,
          3, 100, 0, 25, 69, 108, 105, 120, 105, 114, 46, 78, 120, 46, 67, 111, 110, 116, 97, 105,
          110, 101, 114, 46, 84, 117, 112, 108, 101, 108, 0, 0, 0, 2, 104, 2, 106, 104, 5, 100, 0,
          6, 116, 101, 110, 115, 111, 114, 104, 0, 104, 2, 100, 0, 1, 115, 97, 64, 106, 109, 0, 0,
          0, 8, 1, 0, 0, 0, 0, 0, 0, 0, 104, 2, 106, 104, 5, 100, 0, 6, 116, 101, 110, 115, 111,
          114, 104, 0, 104, 2, 100, 0, 1, 115, 97, 64, 106, 109, 0, 0, 0, 8, 2, 0, 0, 0, 0, 0, 0,
          0, 106, 100, 0, 2, 111, 107, 104, 2, 100, 0, 1, 98, 104, 3, 100, 0, 23, 69, 108, 105,
          120, 105, 114, 46, 78, 120, 46, 67, 111, 110, 116, 97, 105, 110, 101, 114, 46, 77, 97,
          112, 108, 0, 0, 0, 1, 104, 2, 100, 0, 3, 107, 101, 121, 104, 5, 100, 0, 6, 116, 101,
          110, 115, 111, 114, 104, 0, 104, 2, 100, 0, 1, 115, 97, 64, 106, 109, 0, 0, 0, 8, 3, 0,
          0, 0, 0, 0, 0, 0, 106, 100, 0, 2, 111, 107, 106, 108, 0, 0, 0, 1, 104, 2, 100, 0, 1,
          100, 100, 0, 4, 107, 101, 112, 116, 106>>
      ]

      assert {1, :little, _} =
               serialized |> :erlang.iolist_to_binary() |> :erlang.binary_to_term()

      assert Nx.deserialize(serialized) == %Container{
               a: {Nx.tensor(1, type: :s64), Nx.tensor(2, type: :s64)},
               b: %{key: Nx.tensor(3, type: :s64)},
               c: %{},
               d: :kept
             }
    end

    test "containers" do
      container = %Container{
        a: {Nx.tensor(1), Nx.tensor(2)},
        b: %{key: Nx.tensor(3)},
        c: :discarded,
        d: :kept
      }

      assert Nx.deserialize(Nx.serialize(container)) == %{container | c: %{}}
    end

    test "numbers" do
      assert Nx.deserialize(Nx.serialize(123)) == Nx.tensor(123)
      assert Nx.deserialize(Nx.serialize(1.2)) == Nx.tensor(1.2)
    end

    test "vectorized" do
      tensor = Nx.tensor([[1], [2], [3]]) |> Nx.vectorize(:x)
      assert Nx.deserialize(Nx.serialize(tensor)) == tensor
    end
  end

  describe "load_numpy/1" do
    test "loads array" do
      assert Nx.load_numpy!(File.read!("test/fixtures/numpy/no_dims_int64.npy")) ==
               Nx.tensor(123, type: {:s, 64})

      assert Nx.load_numpy!(File.read!("test/fixtures/numpy/1d_int64.npy")) ==
               Nx.tensor([1, 2, 3, 4], type: {:s, 64})

      assert Nx.load_numpy!(File.read!("test/fixtures/numpy/2d_float32.npy")) ==
               Nx.tensor([[1, 2], [3, 4], [5, 6]], type: {:f, 32})

      assert Nx.load_numpy!(File.read!("test/fixtures/numpy/1d_uint8.npy")) ==
               Nx.tensor([1, 2, 3], type: {:u, 8})
    end
  end

  describe "load_numpy_archive!/1" do
    test "loads archive" do
      expected = [
        {"foo", Nx.tensor([1, 2, 3], type: {:s, 64})},
        {"bar", Nx.tensor([-1, 1], type: {:f, 32})},
        {"a_longer_name", Nx.tensor([123], type: {:s, 64})}
      ]

      assert Nx.load_numpy_archive!(File.read!("test/fixtures/numpy/archive.npz")) == expected

      assert Nx.load_numpy_archive!(File.read!("test/fixtures/numpy/archive_compressed.npz")) ==
               expected
    end
  end

  describe "sigils" do
    test "evaluates to tensor" do
      import Nx

      assert ~MAT[-1 2 3 4] == Nx.tensor([[-1, 2, 3, 4]])

      assert ~MAT[1
                2
                3
                4] == Nx.tensor([[1], [2], [3], [4]])

      assert ~MAT[1.0 2  3
                11  12 13] == Nx.tensor([[1.0, 2, 3], [11, 12, 13]])

      assert ~VEC[4 3 2 1] == Nx.tensor([4, 3, 2, 1])
    end

    test "raises when vector has more than one dimension" do
      assert_raise(
        ArgumentError,
        "must be one-dimensional",
        fn ->
          eval(~S[~VEC<0 0 0 1
                     1 0 0 0>])
        end
      )
    end

    test "evaluates with proper type" do
      assert eval("~MAT[1 2 3 4]f32") == Nx.tensor([[1, 2, 3, 4]], type: {:f, 32})
      assert eval("~MAT[4 3 2 1]u8") == Nx.tensor([[4, 3, 2, 1]], type: {:u, 8})

      assert eval("~VEC[0 1 0 1]u8") == Nx.tensor([0, 1, 0, 1], type: {:u, 8})
    end

    test "raises on invalid type" do
      assert_raise(
        ArgumentError,
        "invalid numerical type: {:f, 4} (see Nx.Type docs for all supported types)",
        fn ->
          eval("~MAT[1 2 3 4]f4")
        end
      )
    end

    test "raises on non-numerical values" do
      assert_raise(
        ArgumentError,
        "expected a numerical value for tensor, got x",
        fn ->
          eval("~VEC[1 2 x 4]u8")
        end
      )
    end

    defp eval(expression) do
      "import Nx; #{expression}"
      |> Code.eval_string()
      |> elem(0)
    end
  end

  describe "gather" do
    test "raises when out of bounds" do
      t = Nx.tensor([[1, 2], [3, 4]])

      assert_raise ArgumentError, "index 3 is out of bounds for axis 0 in shape {2, 2}", fn ->
        Nx.gather(t, Nx.tensor([[3, -10]]))
      end

      assert_raise ArgumentError, "index -1 is out of bounds for axis 1 in shape {2, 2}", fn ->
        Nx.gather(t, Nx.tensor([[0, -1]]))
      end
    end
  end

  describe "variance/1" do
    test "calculates variance of a tensor" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])
      assert Nx.variance(t) == Nx.tensor(2.9166667461395264)
    end

    test "uses optional ddof" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])
      assert Nx.variance(t, ddof: 1) == Nx.tensor(3.5)
    end

    test "uses optional axes" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]], names: [:x, :y])

      assert Nx.variance(t, axes: [:x]) ==
               Nx.tensor([1.5555557012557983, 4.222222328186035], names: [:y])

      t = Nx.tensor([[4, 5], [2, 3], [1, 0]], names: [:x, :y])
      assert Nx.variance(t, axes: [:y]) == Nx.tensor([0.25, 0.25, 0.25], names: [:x])
    end

    test "uses optional keep axes" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])
      assert Nx.variance(t, keep_axes: true) == Nx.tensor([[2.9166667461395264]])
    end
  end

  describe "standard_deviation/1" do
    test "calculates the standard deviation of a tensor" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])
      assert Nx.standard_deviation(t) == Nx.tensor(1.707825127659933)
    end

    test "uses optional ddof" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])
      assert Nx.standard_deviation(t, ddof: 1) == Nx.tensor(1.8708287477493286)
    end

    test "uses optional axes" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])

      assert Nx.standard_deviation(t, axes: [0]) ==
               Nx.tensor([1.247219204902649, 2.054804801940918])
    end

    test "uses optional keep axes" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])
      assert Nx.standard_deviation(t, keep_axes: true) == Nx.tensor([[1.7078251838684082]])
    end
  end

  describe "covariance/1" do
    test "calculates the covariance of a tensor" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])

      assert Nx.covariance(t) ==
               Nx.tensor([
                 [1.5555554628372192, 2.444444417953491],
                 [2.444444417953491, 4.222222328186035]
               ])
    end

    test "with tensor batch" do
      t = Nx.tensor([[[4, 5], [2, 3], [1, 0]], [[10, 11], [8, 9], [7, 6]]])

      assert Nx.covariance(t) ==
               Nx.tensor([
                 [
                   [1.5555554628372192, 2.444444417953491],
                   [2.444444417953491, 4.222222328186035]
                 ],
                 [
                   [1.5555554628372192, 2.444444417953491],
                   [2.444444417953491, 4.222222328186035]
                 ]
               ])
    end

    test "with mean" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])
      mean = Nx.tensor([3, 2])

      assert Nx.covariance(t, mean) ==
               Nx.tensor([
                 [2.0, 2.0],
                 [2.0, 4.666666507720947]
               ])
    end

    test "with mean batch" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])
      mean = Nx.tensor([[3, 2], [2, 3]])

      assert Nx.covariance(t, mean) ==
               Nx.tensor([
                 [
                   [2.0, 2.0],
                   [2.0, 4.666666507720947]
                 ],
                 [
                   [1.6666666269302368, 2.3333332538604736],
                   [2.3333332538604736, 4.333333492279053]
                 ]
               ])
    end

    test "uses optional ddof" do
      t = Nx.tensor([[4, 5], [2, 3], [1, 0]])

      assert Nx.covariance(t, ddof: 1) ==
               Nx.tensor([
                 [2.3333332538604736, 3.6666667461395264],
                 [3.6666667461395264, 6.333333492279053]
               ])
    end
  end

  describe "weighted_mean/3" do
    test "shape of input differs from shape of weights" do
      t = Nx.iota({4, 6})
      weights = Nx.tensor([1, 2, 3, 4])

      assert Nx.weighted_mean(t, weights, axes: [0]) ==
               Nx.tensor([12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    end
  end

  describe "median/2" do
    test "uses optional keep axis with unsorted tensor" do
      t = Nx.tensor([[[4, 5, 8], [1, 6, 7]], [[3, 9, 2], [7, 5, 6]]])

      assert Nx.median(t, axis: -1, keep_axis: true) ==
               Nx.tensor([[[5], [6]], [[3], [6]]])
    end
  end

  describe "complex/2" do
    @t32 Nx.tensor(1, type: :f32)
    @t64 Nx.tensor(2, type: :f64)

    test "preserves precision of real and imaginary parts" do
      t = Nx.tensor(Complex.new(1, 2), type: :c128)
      c = Nx.complex(Nx.real(t), Nx.imag(t))
      assert Nx.type(c) == {:c, 128}

      t = Nx.tensor(Complex.new(1, 3), type: :c64)
      c = Nx.complex(Nx.real(t), Nx.imag(t))
      assert Nx.type(c) == {:c, 64}
    end

    test "returns a tensor with type :c128 for 64 bit inputs" do
      t = Nx.complex(@t64, @t64)
      assert Nx.type(t) == {:c, 128}
    end

    test "returns a tensor with type :c128 for mixed 32 and 64 bit inputs" do
      t = Nx.complex(@t32, @t64)
      assert Nx.type(t) == {:c, 128}

      t = Nx.complex(@t64, @t32)
      assert Nx.type(t) == {:c, 128}
    end
  end

  describe "as_type/2" do
    test "preserves the imaginary part when casting between :c68 and :c128" do
      t64 = Nx.tensor(Complex.new(1, 2), type: :c64)
      t128 = Nx.tensor(Complex.new(1, 2), type: :c128)

      assert Nx.as_type(t128, :c64) == t64
      assert Nx.as_type(t64, :c128) == t128
    end

    test "keeps only real part when downcasting complex to real" do
      z = Complex.new(1, 2)
      t64 = Nx.tensor(z, type: :c64)

      for type <- [f: 32, f: 64] do
        assert Nx.as_type(t64, type) == Nx.tensor(z.re, type: type)
      end

      for type <- [s: 64, s: 32, s: 16, s: 8, u: 64, u: 32, u: 16, u: 8] do
        assert Nx.as_type(t64, type) == Nx.tensor(trunc(z.re), type: type)
      end
    end
  end

  describe "with_default_backend/2" do
    test "implicitly uses the given backend inside the given function" do
      left =
        Nx.with_default_backend(Nx.Defn.Expr, fn ->
          Nx.iota({4}) |> Nx.sin()
        end)

      right = Nx.iota({4}, backend: Nx.Defn.Expr) |> Nx.sin()

      assert inspect(left) == inspect(right)
    end
  end

  describe "linspace/3" do
    test "works for :f64 (bug fix: https://github.com/elixir-nx/nx/issues/1222)" do
      linear = Nx.linspace(0.0, 0.5, n: 6, type: :f64)
      expected_linear = Nx.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], type: :f64)
      assert_all_close(linear, expected_linear, atol: 1.0e-15, rtol: 1.0e-15)
    end
  end

  describe "reflect/2" do
    test "reflects over axes of size 1" do
      assert Nx.tensor([1, 1, 1, 1]) == Nx.reflect(Nx.tensor([1]), padding_config: [{2, 1}])

      assert Nx.tensor([[[1, 1, 1, 1]]]) ==
               Nx.reflect(Nx.tensor([[[1]]]), padding_config: [{0, 0}, {0, 0}, {2, 1}])
    end

    test "semantic error on invalid padding config" do
      assert_raise ArgumentError,
                   "expected padding config for axis 0 to be of the format {left, right}, with left and right as non-negative integers, got: {0}",
                   fn ->
                     Nx.reflect(Nx.tensor([0]), padding_config: [{0}])
                   end

      assert_raise ArgumentError,
                   "expected padding config for axis 2 to be of the format {left, right}, with left and right as non-negative integers, got: {-1, 0}",
                   fn ->
                     Nx.reflect(Nx.tensor([[[0]]]), padding_config: [{0, 0}, {0, 0}, {-1, 0}])
                   end
    end
  end

  describe "mode" do
    test "axis" do
      assert Nx.tensor([[1, 2, 3]]) |> Nx.mode(axis: 0) == Nx.tensor([1, 2, 3])
      assert Nx.tensor([[1, 2, 3]]) |> Nx.mode(axis: 0, keep_axis: true) == Nx.tensor([[1, 2, 3]])
      assert Nx.mode(Nx.tensor([[1], [2], [1], [3]]), axis: 1) == Nx.tensor([1, 2, 1, 3])
    end

    test "keep_axis" do
      assert Nx.mode(Nx.tensor(1), keep_axis: true) == Nx.tensor([1])
      assert Nx.mode(Nx.tensor([[[1]]]), keep_axis: true) == Nx.tensor([[[1]]])
    end
  end

  describe "reshape_vectors" do
    test "returns correct axes' order" do
      x = Nx.iota({1}, vectorized_axes: [x: 1])
      xy = Nx.iota({1}, vectorized_axes: [x: 1, y: 2])
      x2 = Nx.iota({1}, vectorized_axes: [x: 2])

      assert [
               %{x | vectorized_axes: [x: 1, y: 1]},
               %{xy | vectorized_axes: [x: 1, y: 2]},
               %{x2 | vectorized_axes: [x: 2, y: 1]}
             ] == Nx.reshape_vectors([x, xy, x2])
    end
  end

  describe "broadcast_vectors" do
    test "returns correct axes' order" do
      x = Nx.iota({1}, vectorized_axes: [x: 1])
      xy = Nx.iota({1, 2}, vectorized_axes: [x: 1, y: 2])
      x2 = Nx.iota({1, 2, 3}, vectorized_axes: [x: 2])

      result = [
        Nx.iota({1}, vectorized_axes: [x: 2, y: 2]),
        Nx.iota({1, 2}, vectorized_axes: [x: 2, y: 2]),
        Nx.iota({1, 2, 3}, vectorized_axes: [x: 2, y: 2])
      ]

      assert result == Nx.broadcast_vectors([x, xy, x2])
      assert result == Nx.broadcast_vectors([x, xy, x2], align_ranks: false)
    end

    test "returns correct axes' order and shape for align_ranks: true" do
      x = Nx.iota({1}, vectorized_axes: [x: 1])
      xy = Nx.iota({1, 2}, vectorized_axes: [x: 1, y: 2])
      x2 = Nx.iota({1, 2, 3}, vectorized_axes: [x: 2])

      assert [
               Nx.iota({1, 1, 1}, vectorized_axes: [x: 2, y: 2]),
               Nx.iota({1, 1, 2}, vectorized_axes: [x: 2, y: 2]),
               Nx.iota({1, 2, 3}, vectorized_axes: [x: 2, y: 2])
             ] == Nx.broadcast_vectors([x, xy, x2], align_ranks: true)
    end
  end

  describe "vectorization" do
    test "vectorizes and devectorizes containers" do
      container = {Nx.tensor([1]), %{a: Nx.tensor([2]), b: {Nx.tensor([3]), Nx.tensor([[4, 5]])}}}

      vectorized = Nx.vectorize(container, x: 1)

      assert vectorized ==
               {Nx.tensor([1]) |> Nx.vectorize(x: 1),
                %{
                  a: Nx.tensor([2]) |> Nx.vectorize(x: 1),
                  b:
                    {Nx.tensor([3]) |> Nx.vectorize(x: 1),
                     Nx.tensor([[4, 5]]) |> Nx.vectorize(x: 1)}
                }}

      assert Nx.devectorize({1, vectorized}, keep_names: false) == {Nx.tensor(1), container}
    end

    test "broadcasts work for vectorized scalars vs non-vectorized operand" do
      v = Nx.tensor([0, 1]) |> Nx.vectorize(x: 2)
      t = Nx.iota({2, 3})

      result =
        Nx.tensor([
          [
            [0, 1, 2],
            [3, 4, 5]
          ],
          [
            [1, 2, 3],
            [4, 5, 6]
          ]
        ])
        |> Nx.vectorize(x: 2)

      assert Nx.add(t, v) == result
      assert Nx.add(v, t) == result
    end

    test "broadcasts work for vectorized 1D tensors vs non-vectorized operand" do
      v = Nx.tensor([[0, 1, 2], [10, 20, 30]]) |> Nx.vectorize(x: 2)
      t = Nx.iota({2, 3})

      result =
        Nx.tensor([
          [
            [0, 2, 4],
            [3, 5, 7]
          ],
          [
            [10, 21, 32],
            [13, 24, 35]
          ]
        ])
        |> Nx.vectorize(x: 2)

      assert Nx.add(t, v) == result
      assert Nx.add(v, t) == result
    end

    test "broadcasts work for vectorized scalars vs vectorized operand" do
      v = Nx.tensor([0, 1]) |> Nx.vectorize(x: 2)
      t = Nx.iota({2, 3}, vectorized_axes: [x: 1, y: 2])

      result =
        Nx.tensor([
          [
            [
              [0, 1, 2],
              [3, 4, 5]
            ],
            [
              [0, 1, 2],
              [3, 4, 5]
            ]
          ],
          [
            [
              [1, 2, 3],
              [4, 5, 6]
            ],
            [
              [1, 2, 3],
              [4, 5, 6]
            ]
          ]
        ])
        |> Nx.vectorize(x: 2, y: 2)

      assert Nx.add(t, v) == result
      assert Nx.add(v, t) == result
    end

    test "broadcasts work for vectorized 1D tensors vs vectorized operand" do
      v = Nx.tensor([[0, 1, 2], [10, 20, 30]]) |> Nx.vectorize(x: 2)
      t = Nx.iota({2, 3}, vectorized_axes: [x: 1, y: 2])

      result =
        Nx.tensor([
          [
            [
              [0, 2, 4],
              [3, 5, 7]
            ],
            [
              [0, 2, 4],
              [3, 5, 7]
            ]
          ],
          [
            [
              [10, 21, 32],
              [13, 24, 35]
            ],
            [
              [10, 21, 32],
              [13, 24, 35]
            ]
          ]
        ])
        |> Nx.vectorize(x: 2, y: 2)

      assert Nx.add(t, v) == result
      assert Nx.add(v, t) == result
    end
  end

  describe "select" do
    test "works with vectorized scalar pred" do
      pred = Nx.tensor([0, 1]) |> Nx.vectorize(x: 2)
      on_true = Nx.tensor([[0, 1, 2]])
      on_false = Nx.tensor([[3, 4, 5]])

      assert Nx.select(pred, on_true, on_false) ==
               Nx.tensor([[[3, 4, 5]], [[0, 1, 2]]]) |> Nx.vectorize(x: 2)

      pred = Nx.tensor([0, 1]) |> Nx.vectorize(x: 2)
      on_true = Nx.tensor([[0, 1, 2]]) |> Nx.vectorize(x: 1)
      on_false = Nx.tensor([[3, 4, 5]]) |> Nx.vectorize(x: 1)

      assert Nx.select(pred, on_true, on_false) ==
               Nx.tensor([[3, 4, 5], [0, 1, 2]]) |> Nx.vectorize(x: 2)
    end

    test "works with scalar pred and vectorized inputs" do
      assert Nx.select(
               1,
               Nx.iota({2}, vectorized_axes: [x: 2]),
               Nx.tensor([[3, 4]]) |> Nx.vectorize(x: 1)
             ) == Nx.tensor([[0, 1], [0, 1]]) |> Nx.vectorize(x: 2)
    end
  end

  describe "fft/ifft" do
    test "Bluestein clause" do
      # we need a tensor which isn't a power of 2 and is > than 1024 to actually
      # validate the bluestein implementation. No doctests check for it.
      x = Nx.tile(~VEC[1 1 1 0 0 0], [400])
      x_fft = Nx.fft(x)

      # From numpy we expect the following indices to be the only non-zero values in the tensor
      non_zero_idx = Nx.tensor([0, 400, 1200, 2000])
      assert Nx.take(x_fft, non_zero_idx) == ~VEC[1200 400-692.820323i 400 400+692.820323i]

      zeros_check = Nx.indexed_put(x_fft, Nx.new_axis(non_zero_idx, 1), ~VEC[0 0 0 0])
      zeros = Nx.broadcast(0, x_fft)
      assert_all_close(Nx.real(zeros_check), zeros, atol: 1.0e-9)
      assert_all_close(Nx.imag(zeros_check), zeros, atol: 1.0e-9)

      # Because we implement ifft in terms of fft, we'll use the same tensor as validation
      x_ifft = Nx.ifft(x_fft)
      assert_all_close(x, Nx.real(x_ifft), atol: 1.0e-8)
      assert_all_close(zeros, Nx.imag(x_ifft), atol: 1.0e-8)
    end
  end

  describe "split/2" do
    test "split is less than zero" do
      tensor = Nx.iota({10, 3}, names: [:x, :y])

      assert Nx.split(tensor, -1, axis: 1) == {tensor[[.., 0..1]], tensor[[.., 2..2]]}
    end

    test "split with negative float" do
      tensor = Nx.iota({10, 2}, names: [:x, :y])

      assert_raise ArgumentError,
                   "split must be a float such that 0 < split and ceil(split * axis_size) < 1",
                   fn ->
                     Nx.split(tensor, -0.5)
                   end
    end

    test "split is a float out of bounds" do
      tensor = Nx.iota({10, 2}, names: [:x, :y])

      assert_raise ArgumentError,
                   "split must be a float such that 0 < split and ceil(split * axis_size) < 1",
                   fn ->
                     Nx.split(tensor, 1.0)
                   end
    end

    test "split is greater than tensor length" do
      tensor = Nx.iota({10, 2}, names: [:x, :y])

      assert_raise ArgumentError,
                   "split must be an integer greater than zero and less than the length of the given axis",
                   fn ->
                     Nx.split(tensor, 3, axis: 1)
                   end
    end

    test "axis is out of tensor bounds" do
      tensor = Nx.iota({10, 2}, names: [:x, :y])

      assert_raise ArgumentError,
                   "given axis (2) invalid for shape with rank 2",
                   fn ->
                     Nx.split(tensor, 2, axis: 2)
                   end
    end

    test "named axis is invalid" do
      tensor = Nx.iota({10, 2}, names: [:x, :y])

      assert_raise ArgumentError,
                   "name :z not found in tensor with names [:x, :y]",
                   fn ->
                     Nx.split(tensor, 2, axis: :z)
                   end
    end

    test "split into 50% for training and 50% for testing with floats on columns" do
      tensor = Nx.iota({4, 4}, names: [:rows, :columns])
      {train, test} = Nx.split(tensor, 0.5, axis: :columns)

      assert {4, 2} == Nx.shape(train)
      assert {4, 2} == Nx.shape(test)
    end

    test "split into 70% for training and 30% for testing along a named :axis" do
      tensor = Nx.iota({100, 6}, names: [:rows, :columns])
      {train, test} = Nx.split(tensor, 70, axis: :rows)

      assert {70, 6} == Nx.shape(train)
      assert {30, 6} == Nx.shape(test)
    end

    test "split into 90% for training and 10% for testing along a named :axis" do
      tensor = Nx.iota({2, 100}, names: [:rows, :columns])
      {train, test} = Nx.split(tensor, 90, axis: :columns)

      assert {2, 90} == Nx.shape(train)
      assert {2, 10} == Nx.shape(test)
    end

    test "split into 50% for training and 50% for testing along the :axis 1" do
      tensor = Nx.iota({100, 10})
      {train, test} = Nx.split(tensor, 5, axis: 1)

      assert {100, 5} == Nx.shape(train)
      assert {100, 5} == Nx.shape(test)
    end

    test "split into 61% for training and 39% for testing" do
      tensor = Nx.iota({100, 10})
      {train, test} = Nx.split(tensor, 61)

      assert {61, 10} == Nx.shape(train)
      assert {39, 10} == Nx.shape(test)
    end

    test "split into 60% for training and 40% for testing with unbalanced data" do
      tensor = Nx.iota({99, 4})

      {train, test} = Nx.split(tensor, 60)

      assert {60, 4} == Nx.shape(train)
      assert {39, 4} == Nx.shape(test)
    end
  end

  describe "logsumexp" do
    test "removes the right axes" do
      tensor = Nx.tensor([[[11.0], [12.0], [13.0]]])

      assert Nx.logsumexp(tensor, axes: [0]) ==
               Nx.tensor([
                 [11.0],
                 [12.0],
                 [13.0]
               ])

      assert Nx.logsumexp(tensor, axes: [2]) ==
               Nx.tensor([
                 [11.0, 12.0, 13.0]
               ])
    end

    test "works with infinities" do
      tensor =
        Nx.tensor([
          [:infinity, :infinity, :infinity],
          [:infinity, :infinity, :neg_infinity],
          [:infinity, :infinity, 0.0],
          [:infinity, :neg_infinity, :infinity],
          [:infinity, :neg_infinity, :neg_infinity],
          [:infinity, :neg_infinity, 0.0],
          [:infinity, 0.0, :infinity],
          [:infinity, 0.0, :neg_infinity],
          [:infinity, 0.0, 0.0],
          [:neg_infinity, :infinity, :infinity],
          [:neg_infinity, :infinity, :neg_infinity],
          [:neg_infinity, :infinity, 0.0],
          [:neg_infinity, :neg_infinity, :infinity],
          [:neg_infinity, :neg_infinity, :neg_infinity],
          [:neg_infinity, :neg_infinity, 0.0],
          [:neg_infinity, 0.0, :infinity],
          [:neg_infinity, 0.0, :neg_infinity],
          [:neg_infinity, 0.0, 0.0],
          [0.0, :infinity, :infinity],
          [0.0, :infinity, :neg_infinity],
          [0.0, :infinity, 0.0],
          [0.0, :neg_infinity, :infinity],
          [0.0, :neg_infinity, :neg_infinity],
          [0.0, :neg_infinity, 0.0],
          [0.0, 0.0, :infinity],
          [0.0, 0.0, :neg_infinity]
        ])

      assert Nx.logsumexp(tensor, axes: [1]) ==
               Nx.tensor([
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 :neg_infinity,
                 0.0,
                 :infinity,
                 0.0,
                 0.6931471824645996,
                 :infinity,
                 :infinity,
                 :infinity,
                 :infinity,
                 0.0,
                 0.6931471824645996,
                 :infinity,
                 0.6931471824645996
               ])
    end
  end

  describe "quantized types" do
    test "s2" do
      tensor = Nx.s2([-2, -1, 1])
      assert tensor.type == {:s, 2}

      assert <<-2::2-signed-native, -1::2-signed-native, 1::2-signed-native>> =
               Nx.to_binary(tensor)

      assert [-2, -1, 1] = Nx.to_flat_list(tensor)
      assert 0 = Nx.byte_size(tensor)
      assert 6 = Nx.bit_size(tensor)

      tensor = Nx.s2([-2, -1, 0, 1, 0, -1, -2])
      assert 1 = Nx.byte_size(tensor)
      assert 14 = Nx.bit_size(tensor)
    end

    test "s4" do
      tensor = Nx.s4([-8, -1, 7])
      assert tensor.type == {:s, 4}

      assert <<-8::4-signed-native, -1::4-signed-native, 7::4-signed-native>> =
               Nx.to_binary(tensor)

      assert [-8, -1, 7] = Nx.to_flat_list(tensor)
      assert 1 = Nx.byte_size(tensor)
      assert 12 = Nx.bit_size(tensor)

      tensor = Nx.s4([-8, -3, 0, 7, 0, -3, -8])
      assert 3 = Nx.byte_size(tensor)
      assert 28 = Nx.bit_size(tensor)
    end

    test "u2" do
      tensor = Nx.u2([1, 2, 3])
      assert tensor.type == {:u, 2}
      assert <<1::2-native, 2::2-native, 3::2-native>> = Nx.to_binary(tensor)
      assert [1, 2, 3] = Nx.to_flat_list(tensor)
      assert 0 = Nx.byte_size(tensor)
      assert 6 = Nx.bit_size(tensor)

      tensor = Nx.u2([0, 1, 2, 3, 2, 1, 0])
      assert 1 = Nx.byte_size(tensor)
      assert 14 = Nx.bit_size(tensor)
    end

    test "u4" do
      tensor = Nx.u4([0, 7, 15])
      assert tensor.type == {:u, 4}
      assert <<0::4-native, 7::4-native, 15::4-native>> = Nx.to_binary(tensor)
      assert [0, 7, 15] = Nx.to_flat_list(tensor)
      assert 1 = Nx.byte_size(tensor)
      assert 12 = Nx.bit_size(tensor)

      tensor = Nx.u4([0, 1, 2, 3, 13, 14, 15])
      assert 3 = Nx.byte_size(tensor)
      assert 28 = Nx.bit_size(tensor)
    end
  end
end
